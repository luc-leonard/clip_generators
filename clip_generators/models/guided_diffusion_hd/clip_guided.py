import os
import os
import time
import urllib.request
from pathlib import Path
from typing import io as typing_io, List, Tuple, Optional
import io
import clip
import imageio
import numpy as np
import requests
import torch
import PIL.Image
from progressbar import progressbar
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from clip_generators.models.guided_diffusion_hd.guided_diffusion.guided_diffusion.script_util import \
    create_model_and_diffusion, model_and_diffusion_defaults
from clip_generators.models.guided_diffusion_hd.discriminator import ClipDiscriminator
from clip_generators.utils import fetch

import gc


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def make_model(the_model_config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model, diffusion = create_model_and_diffusion(**the_model_config)
    current_path = os.path.dirname(os.path.realpath(__file__))

    checkpoint_path = Path(current_path) / '512x512_diffusion_uncond_finetune_008100.pt'
    if not checkpoint_path.exists():
        urllib.request.URLopener().retrieve(
            'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt',
            str(checkpoint_path), reporthook=lambda block_id, bs, size: print(block_id, bs, size))
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if the_model_config['use_fp16']:
        model.convert_to_fp16()

    return model, diffusion


normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])


def generate(prompts: List[Tuple[str, float]],
             clip_model,
             out_dir: Path,
             ddim_respacing: bool = True,
             init_image_url: Optional[str] = None, seed=None, steps=1000, skip_timesteps=0):
    batch_size = 1
    clip_guidance_scale = 1000
    tv_scale = 150
    cutn = 40
    cut_pow = 0.5
    n_batches = 1

    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': steps,
        'rescale_timesteps': True,
        'timestep_respacing': 'ddim' + str(steps) if ddim_respacing else str(steps),
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })

    model, diffusion = make_model(model_config)

    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(time.time())

    discriminator = ClipDiscriminator(clip_model, prompts, cutn, cut_pow, 'cuda:0', False, 0)
    init = None
    print(init_image_url)
    image_size = model_config['image_size']
    if init_image_url is not None:
        init_image: PIL.Image.Image = PIL.Image.open(fetch(init_image_url)).convert('RGB')
        init_image.thumbnail((image_size, image_size))
        init_image.save(f'./{str(out_dir)}/progress_latest.png')
        print('saved init')
        yield -1
        resized_init_image = PIL.Image.new('RGB', (image_size, image_size), color=1)

        resized_init_image.paste(init)
        resized_init_image = resized_init_image.resize((model_config['image_size'], model_config['image_size']), PIL.Image.LANCZOS)

        init = TF.to_tensor(resized_init_image).to('cuda:0').unsqueeze(0).mul(2).sub(1)
        init: torch.Tensor = (init * (torch.rand_like(init) * 0.5))

    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device='cuda:0', dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)

            dists = discriminator(x_in.add(1).div(2), n)

            losses = torch.cat(dists).mean()
            tv_losses = tv_loss(x_in)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
            return -torch.autograd.grad(loss, x)[0]

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
        video = imageio.get_writer(f'{out_dir}/out.mp4', mode='I', fps=5, codec='libx264', bitrate='16M')
        for j, sample in progressbar(enumerate(samples)):
            cur_t -= 1

            for k, image in enumerate(sample['pred_xstart']):
                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                if j % 25 == 0 or cur_t == -1:
                    video.append_data(np.array(image))
                if j % 100 == 0 or cur_t == -1:
                    image.save(f'./{str(out_dir)}/progress_latest.png')
                yield j + skip_timesteps
        video.close()
        del model
        gc.collect()


class Dreamer:
    def __init__(self, prompts, clip_model, *, outdir: str, init_image: Optional[str], ddim_respacing, seed, steps, skip_timesteps):
        self.prompts = prompts
        self.clip = clip_model
        self.out_dir = Path(outdir)
        self.init_image = init_image
        self.ddim_respacing = ddim_respacing
        self.seed = seed
        self.steps = steps
        self.skip_timesteps = skip_timesteps

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def epoch(self):
        return generate(self.prompts, self.clip, self.out_dir,
                        init_image_url=self.init_image,
                        ddim_respacing=self.ddim_respacing,
                        seed=self.seed,
                        steps=self.steps, skip_timesteps=self.skip_timesteps)

    def get_generated_image_path(self) -> Path:
        return self.out_dir / 'progress_latest.png'

    def close(self):
        ...

    @property
    def prompt(self):
        return self.prompts[0][0]

