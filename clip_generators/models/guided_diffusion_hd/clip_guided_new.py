import gc
import io
import math
import os
import sys
import urllib
from pathlib import Path
from typing import List, Optional

import imageio
import lpips
from PIL import Image, ImageOps
import requests
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import transforms
from tqdm.notebook import tqdm


import clip
from clip_generators.models.guided_diffusion_hd.guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import random

from clip_generators.models.upscaler.upscaler import upscale

device = 'cuda:0'


# 350/50/50/32 and 500/0/0/64 have worked well for 25 timesteps on 256px
# Also, sometimes 1 cutn actually works out fine

prompts = ['princess in sanctuary, trending on artstation, photorealistic portrait of a young princess']
image_prompts = []
batch_size = 1
clip_guidance_scale = 5000  # Controls how much the image should look like the prompt. Use high value when clamping activated
tv_scale = 8000              # Controls the smoothness of the final output.
range_scale = 150            # Controls how far out of range RGB values are allowed to be.
clamp_max=0.5              # Controls how far gradient can go - try play with it, dramatic effect when clip guidance scale is high enough

RGB_min, RGB_max = [-0.9,0.9]     # Play with it to get different styles
cutn = 32
cutn_batches = 4           # Turn this up for better result but slower speed
cutn_whole_portion = 0.2       #The rotation augmentation, captures whole structure
rotation_fill=[1,1,1]
cutn_bw_portion = 0.2         #Greyscale augmentation, focus on structure rather than color info to give better structure
cut_pow = 0.5
n_batches = 1
#init_image = None   # This can be an URL or Colab local path and must be in quotes.
skip_timesteps = 12  # Skip unstable steps                  # Higher values make the output look more like the init.
init_scale = 1000      # This enhances the effect of the init image, a good value is 1000.
clip_denoised = False

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True, side_x=512, side_y=512):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_x, side_y), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_x, side_y), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., cutn_whole_portion = 0.0, cutn_bw_portion = 0.2):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.cutn_whole_portion = cutn_whole_portion
        self.cutn_bw_portion = cutn_bw_portion

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        if self.cutn==1:
            cutouts.append(F.adaptive_avg_pool2d(input, self.cut_size))
            return torch.cat(cutouts)
        cut_1 = round(self.cutn*(1-self.cutn_bw_portion))
        cut_2 = self.cutn-cut_1
        gray = transforms.Grayscale(3)
        if cut_1 >0:
            for i in range(cut_1):
                size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i < int(self.cutn_bw_portion * cut_1):
                    cutout = gray(cutout)
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        if cut_2 >0:
            for i in range(cut_2):
                cutout = TF.rotate(input, angle=random.uniform(-10.0, 10.0), expand=True, fill=rotation_fill)
                if i < int(self.cutn_bw_portion * cut_2):
                    cutout =gray(cutout)
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


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


model_list = [
   #'RN50x16',
   "ViT-B/16",
  # "ViT-B/32"
]

def do_run(prompts, clip_model, outdir, seed, init_image):
    #diffusion_steps = 1000

    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': 'ddim100',
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

    side_x = side_y = model_config['image_size']

    model, diffusion = make_model(model_config)

    clip_size = {}
    for i in model_list:
        clip_size[i] = clip_model[i].visual.input_resolution
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    if seed is not None:
        torch.manual_seed(seed)
    make_cutouts = {}
    for i in model_list:
        make_cutouts[i] = MakeCutouts(clip_size[i], cutn // len(model_list), cut_pow, cutn_whole_portion,
                                      cutn_bw_portion)

    side_x = side_y = model_config['image_size']

    target_embeds, weights = {}, []
    for i in model_list:
        target_embeds[i] = []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        for i in model_list:
            target_embeds[i].append(clip_model[i].encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        for i in model_list:
            batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
            embed = clip_model[i].encode_image(normalize(batch)).float()
            target_embeds[i].append(embed)
        weights.extend([weight / cutn * len(model_list)] * (cutn // len(model_list)))
    for i in model_list:
        target_embeds[i] = torch.cat(target_embeds[i])
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, out, y=None):
        clip_guidance_scale_2 = clip_guidance_scale

        n = x.shape[0]
        cur_output = out['pred_xstart'].detach()
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)

        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
        loss = 0
        x_in_grad = torch.zeros_like(x_in)
        for k in range(cutn_batches):
            losses = 0
            for i in model_list:
                if i == "":
                    clip_in = normalize(make_cutouts[i](x_in.mean(dim=1).expand(3, -1, -1).unsqueeze(0).add(1).div(2)))
                else:
                    clip_in = normalize(make_cutouts[i](x_in.add(1).div(2)))
                image_embeds = clip_model[i].encode_image(clip_in).float()
                image_embeds = image_embeds.unsqueeze(1)
                dists = spherical_dist_loss(image_embeds, target_embeds[i].unsqueeze(0))
                del image_embeds, clip_in
                dists = dists.view([cutn // len(model_list), n, -1])
                losses = dists.mul(weights).sum(2).mean(0)
                x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale_2, x_in)[0] / cutn_batches / len(
                    model_list)
                del dists, losses
            gc.collect()
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(out['pred_xstart'])
        loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
        if init is not None and init_scale:
            init_losses = lpips_model(x_in, init)
            loss = loss + init_losses.sum() * init_scale
        x_in_grad += torch.autograd.grad(loss, x_in, )[0]
        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        magnitude = grad.square().mean().sqrt()
        return grad * magnitude.clamp(max=clamp_max) / magnitude

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive
    video = imageio.get_writer(f'{outdir}/out.mp4', mode='I', fps=25, codec='libx264', bitrate='16M')

    for i in range(n_batches):

        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            clip_denoised=clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            cond_fn_with_grad=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            for k, image in enumerate(sample['pred_xstart']):
                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                video.append_data(np.array(image))
                if j % 10 == 0 or cur_t == -1:
                    tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    image.save(str(outdir) + '/progress_latest.png')
                    yield j, str(outdir) + '/progress_latest.png'


class Dreamer:
    def __init__(self,
                 clip_model, *,
                 init_image: Optional[str] = None,
                 seed,):
        self.clip = clip_model
        self.init_image = init_image
        self.seed = seed
        self.steps = 124 #yes it is hardcoded :)


    def type(self):
        return 'diffusion'

    def generate(self, prompt, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return do_run([x[0] for x in prompt], {"ViT-B/16": self.clip}, out_dir, self.seed, self.init_image)

    def upsampler(self):
        return upscale

    def same_arguments(self, _): return False