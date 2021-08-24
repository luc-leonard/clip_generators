import os
import os
import time
import urllib.request
from pathlib import Path
from typing import io

import clip
import imageio
import numpy as np
import torch
from google.auth.transport import requests
from progressbar import progressbar
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from clip_generators.models.guided_diffusion_hd.guided_diffusion.guided_diffusion.script_util import \
    create_model_and_diffusion, model_and_diffusion_defaults
from clip_generators.models.taming_transformers.clip_generator.utils import MakeCutouts


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


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
    if model_config['use_fp16']:
        model.convert_to_fp16()

    return model, diffusion


model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',  # Modify this value to decrease the number of
    # timesteps.
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
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])


def generate(prompt: str, clip_model, out_dir: Path):
    batch_size = 1
    clip_guidance_scale = 1000
    tv_scale = 150
    cutn = 40
    cut_pow = 0.5
    n_batches = 1
    init_image = None
    skip_timesteps = 0
    seed = None

    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(time.time())

    clip_size = clip_model.visual.input_resolution
    text_embed = clip_model.encode_text(clip.tokenize(prompt).to('cuda:0')).float()

    init = None
    # if init_image is not None:
    #     init = Image.open(fetch(init_image)).convert('RGB')
    #     init = init.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
    #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)
    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device='cuda:0', dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            # clip_in = differentiable_augmentation(clip_in)
            image_embeds = clip_model.encode_image(clip_in).float().view([cutn, n, -1])
            dists = spherical_dist_loss(image_embeds, text_embed.unsqueeze(0))
            losses = dists.mean(0)
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
                yield j
        video.close()


class Trainer:
    def __init__(self, prompt, clip_model, *, outdir: str, ):
        self.prompt = prompt
        self.clip = clip_model
        self.prompts = [prompt]
        self.out_dir = Path(outdir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def epoch(self):
        return generate(self.prompt, self.clip, self.out_dir)

    def get_generated_image_path(self) -> Path:
        return self.out_dir / 'progress_latest.png'

    def close(self):
        ...

    # the diffusion model has a fixed amount of steps
    @property
    def steps(self):
        return 1000
