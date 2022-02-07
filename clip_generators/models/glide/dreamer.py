import datetime
import random
import time
from pathlib import Path
from typing import Dict, Union, Any

import PIL.Image
import torch as th
from PIL import Image
from IPython.display import display
import torch as th
import torch.nn as nn
from torchvision.utils import make_grid

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

import clip_generators.models.upscaler.upscaler
from clip_generators.utils import GenerationArgs

device = 'cuda'


def generate_small(model, diffusion, options, clip_model, prompt, batch_size, guidance_scale):
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor([tokens] * batch_size, device=device),
        mask=th.tensor([mask] * batch_size, dtype=th.bool, device=device),
    )

    # Setup guidance function for CLIP model.
    cond_fn = clip_model.cond_fn([prompt] * batch_size, guidance_scale)

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model,
        (batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    model.del_cache()
    # Show the output
    return samples


def upsample(samples, model_up, diffusion_up, options_up, prompt, batch_size):
    upsample_temp = 0.997
    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model_up.del_cache()

    # Show the output
    return up_samples


def save(batch: th.Tensor, out_path: Path, all):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    grid = make_grid(scaled, nrow=9).permute(1, 2, 0)

    if all:
        for i, image in enumerate(scaled):
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(out_path.parent / f"{i}.png")
    Image.fromarray(grid.numpy()).save(out_path)


def generate(model, diffusion, options, up_model, diffusion_up, options_up, clip_model, prompt, batch_size,
             guidance_scale, out_dir):
    samples = generate_small(model, diffusion, options, clip_model, prompt, batch_size, guidance_scale)
    save(samples, out_dir / "samples.png", False)
    yield 1, out_dir / "samples.png"
    upsamples = upsample(samples, up_model, diffusion_up, options_up, prompt, batch_size)
    save(upsamples, out_dir / "upsamples.png", True)
    now = time.time()
    save(upsamples, out_dir / ".." / f'{now}_{prompt}.png', False)
    yield 2, out_dir / "upsamples.png"

class GlideDreamer:
    options: Dict[str, Any]

    def __init__(self, batch_size, clip_guidance, steps, upscale_steps):
        self.batch_size = batch_size
        self.clip_guidance = clip_guidance
        self.steps = steps
        self.upscale_steps = upscale_steps
        self.outdir = None

        self.make_models()

    def same_arguments(self, args: GenerationArgs):
        return self.clip_guidance == args.model_arguments.clip_guidance_scale and self.steps == args.steps and self.upscale_steps == args.model_arguments.upsample_steps

    def make_models(self):
        self.make_generator_model()
        self.make_upscale_model()
        self.make_clip_model()

    def make_clip_model(self):
        clip_model = create_clip_model(device=device)
        clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
        clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))
        self.clip_model = clip_model

    def make_upscale_model(self):
        options_up = model_and_diffusion_defaults_upsampler()
        options_up['use_fp16'] = True
        options_up['timestep_respacing'] = self.upscale_steps  # use 27 diffusion steps for very fast sampling
        model_up, diffusion_up = create_model_and_diffusion(**options_up)
        model_up.eval()
        model_up.convert_to_fp16()
        model_up.to(device)
        model_up.load_state_dict(load_checkpoint('upsample', device))
        print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

        self.model_up = model_up
        self.options_up = options_up
        self.diffusion_up = diffusion_up

    def make_generator_model(self):
        options = model_and_diffusion_defaults()
        options['use_fp16'] = True
        options['timestep_respacing'] = self.steps  # use 100 diffusion steps for fast sampling
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        model.convert_to_fp16()
        model.to(device)
        model.load_state_dict(load_checkpoint('base', device))
        print('total base parameters', sum(x.numel() for x in model.parameters()))

        self.model = model
        self.options = options
        self.diffusion = diffusion

    def generate(self, prompt, out_dir):
        # not threadsafe
        self.outdir = Path(out_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return generate(self.model, self.diffusion, self.options, self.model_up, self.diffusion_up, self.options_up,
                        self.clip_model, prompt[0][0], self.batch_size, self.clip_guidance, out_dir)

    def type(self):
        return "glide"

    def upsampler(self):
        def _upsample(source_path, dest_path):
            source_path = Path(source_path).parent / (str(random.randint(0, 9)) + '.png')

            clip_generators.models.upscaler.upscaler.latent_upscale(str(source_path), dest_path)
        return _upsample

    def close(self):
        ...