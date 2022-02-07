#!/usr/bin/env python3

"""CLIP guided sampling from a diffusion model."""

import argparse
import shutil
import time
from functools import partial
from pathlib import Path

import PIL.Image
import torchvision
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from clip import clip
from clip_generators.models.new_diffusion.diffusion import get_model, get_models, sampling, utils
import kornia.augmentation.augmentation as K

MODULE_DIR = Path(__file__).resolve().parent
from clip_generators.models.upscaler.upscaler import upscale, latent_upscale


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., apply_transforms=True):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        if apply_transforms:
            self.augmentations = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomSharpness(0.3, p=0.4),
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
                K.RandomPerspective(0.2, p=0.4), )
        else:
            self.augmentations = nn.Identity()

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)

            cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        cutouts = self.augmentations(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def sample(prompts=[], images=[], batch_size=1, n=1, seed=0, steps=100, eta=0.0, outdir='.', cutn=64,
           clip_guidance_scale=500,
           apply_transforms=True,
           init_url: str = None,
           skip_steps=0.0,
           size=256,
           model_name='yfcc_2'):
    print('Sampling with ', locals())

    device = 'cuda'
    model = get_model(model_name)()
    if size is None:
        _, side_y, side_x = model.shape
    else:
        side_y, side_x = size, size

    checkpoint = MODULE_DIR / f'checkpoints/{model_name}.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model_name = model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn=cutn, cut_pow=1,
                               apply_transforms=apply_transforms)

    target_embeds, weights = [], []

    if init_url:
        init = Image.open(utils.fetch(init_url)).convert('RGB')
        init = resize_and_center_crop(init, (side_x, side_y))
        init = utils.from_pil_image(init).cuda()[None].repeat([n, 1, 1, 1])

    for prompt in prompts:
        txt, weight = prompt
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in images:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size),
                        transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img)[None].to(device))
        embeds = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embeds)
        weights.extend([weight / cutn] * cutn)

    if not target_embeds:
        raise RuntimeError('At least one text or image prompt must be specified.')
    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    clip_embed = F.normalize(target_embeds.mul(weights[:, None]).sum(0, keepdim=True), dim=-1)
    clip_embed = clip_embed.repeat([n, 1])

    torch.manual_seed(seed)

    def cond_fn(x, t, pred, clip_embed):
        clip_in = normalize(make_cutouts((pred + 1) / 2))
        image_embeds = clip_model.encode_image(clip_in).view([cutn, x.shape[0], -1])
        losses = spherical_dist_loss(image_embeds, clip_embed[None])
        loss = losses.mean(0).sum() * clip_guidance_scale
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    def run(x, steps, clip_embed):
        if hasattr(model, 'clip_model'):
            extra_args = {'clip_embed': clip_embed}
            cond_fn_ = cond_fn
        else:
            extra_args = {}
            cond_fn_ = partial(cond_fn, clip_embed=clip_embed)
        if not clip_guidance_scale:
            return sampling.sample(model, x, steps, eta, extra_args)
        return sampling.cond_sample(model, x, steps, eta, extra_args, cond_fn_)

    def run_all(n, batch_size, _steps):
        x = torch.randn([n, 3, side_y, side_x], device=device)
        # utils.to_pil_image(x).save(f'{outdir}/out_{prompts[0][0]}_init.png')
        # yield -1,  f'{outdir}/out_{prompts[0][0]}_init.png'
        t = torch.linspace(1, 0, _steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        if init_url:
            steps = steps[steps < skip_steps]
            alpha, sigma = utils.t_to_alpha_sigma(steps[0])
            x = init * alpha + x * sigma
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            outs = run(x[i:i + cur_batch_size], steps, clip_embed[i:i + cur_batch_size])
            last_path = f'{outdir}/last.png'

            for j, out in enumerate(outs):
                if j % (_steps / 10) == 0:
                    out_path = f'{outdir}/out_{prompts[0][0]}{i + j:05}.png'
                    if batch_size > 1:
                        out = torchvision.utils.make_grid(out, nrow=2)
                    utils.to_pil_image(out).save(out_path)
                    shutil.copy(out_path, last_path)
                    yield j, out_path
                else:
                    yield None
            yield j, out_path

    try:
        return run_all(n, batch_size, steps)
    except KeyboardInterrupt:
        pass


def cfg_sample(prompts=[], images=[], batch_size=1, n=1, seed=0, steps=100, eta=0.0, outdir='.', cutn=64,
               clip_guidance_scale=500,
               apply_transforms=True,
               init_url: str = None,
               skip_steps=0.0,
               size=256,
               model_name='yfcc_2',
               clip_stop_at=0.75, progress_bar_fn=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = get_model(model_name)()
    _, side_y, side_x = model.shape
    if size:
        side_x, side_y = size
    checkpoint = MODULE_DIR / f'checkpoints/{model_name}.pth'

    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    if device.type == 'cuda':
        model = model.half()

    model = model.to(device).eval().requires_grad_(False)
    clip_model_name = model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn=cutn, cut_pow=1,
                               apply_transforms=apply_transforms)
    if init_url:
        init = Image.open(utils.fetch(init_url)).convert('RGB')
        init = resize_and_center_crop(init, (side_x, side_y))
        init = utils.from_pil_image(init).cuda()[None].repeat([n, 1, 1, 1])

    zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
    target_embeds, weights = [zero_embed], []

    for prompt in prompts:
        txt, weight = prompt
        # weight = weight * 5
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in images:
        path, weight = parse_prompt(prompt)

        img = Image.open(utils.fetch(path)).convert('RGB')
        clip_size = clip_model.visual.input_resolution
        img = resize_and_center_crop(img, (clip_size, clip_size))
        batch = TF.to_tensor(img)[None].to(device)
        embed = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embed)
        weights.append(weight)

    weights = torch.tensor([1 - sum(weights), *weights], device=device)

    torch.manual_seed(seed)
    clip_embed = F.normalize(torch.cat(target_embeds).mul(weights[:, None]).sum(0, keepdim=True), dim=-1)
    clip_embed = clip_embed.repeat([n, 1])

    def cond_fn(x, t, pred, clip_embed):
        if t < clip_stop_at:
            return None
        clip_in = normalize(make_cutouts((pred + 1) / 2))
        image_embeds = clip_model.encode_image(clip_in).view([cutn, x.shape[0], -1])
        losses = spherical_dist_loss(image_embeds, clip_embed[None])
        loss = losses.mean(0).sum() * clip_guidance_scale
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    def cfg_model_fn(x, t, **kwargs):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    def run(x, steps, clip_embed):
        if hasattr(model, 'clip_model'):
            extra_args = {'clip_embed': clip_embed}
        else:
            extra_args = {}
        if not clip_guidance_scale:
            print('No guidance loss')
            return sampling.sample(cfg_model_fn, x, steps, eta, {})
        else:
            return sampling.cond_sample(cfg_model_fn, x, steps, eta, extra_args, cond_fn)

    def run_all(n, _steps, batch_size):
        x = torch.randn([n, 3, side_y, side_x], device=device)
        t = torch.linspace(1, 0, _steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        if init_url:
            steps = steps[steps < skip_steps]
            alpha, sigma = utils.t_to_alpha_sigma(steps[0])
            x = init * alpha + x * sigma
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            outs = run(x[i:i + cur_batch_size], steps, clip_embed[i:i + cur_batch_size])
            for j, out in enumerate(outs):
                if j % (_steps / 10) == 0:
                    out_paths = make_output(batch_size, i, j, out)
                    yield j, out_paths
                else:
                    yield None

            yield j, make_output(batch_size, i, j, out, force_separated_images=True)
            yield j, make_output(batch_size, i, j, out)

    def make_output(batch_size, i, j, out, force_separated_images=False):
        out_paths = []
        if batch_size == 1 or force_separated_images:
            for k, image in enumerate(out):
                out_path = f'{outdir}/out_{prompts[0][0].replace(" ", "_")}_{k}_{i + j:05}.png'
                utils.to_pil_image(image).save(out_path)
                out_paths.append(out_path)
            return out_paths
        if batch_size > 1:
            grid = torchvision.utils.make_grid(out, nrow=2)
            grid_out_path = f'{outdir}/out_{prompts[0][0].replace(" ", "_")}_grid_{i + j:05}.png'
            utils.to_pil_image(grid).save(grid_out_path)
            return [grid_out_path]
        return 'THIS SHOULD NOT HAPPEN'

    try:
        return run_all(n, steps, batch_size)
    except KeyboardInterrupt:
        pass


class NewGenDiffusionDreamer:
    def __init__(self, size=None, images=[], seed=0, steps=100, eta=1.0, outdir='.', cutn=128, clip_guidance_scale=500,
                 transform=True,
                 init=None, skip_steps=0.0, model: str = None, n=None, progress_bar_fn=None):
        self.size = size
        self.images = images
        if n is None:
            self.batch_size = 4 if 'cfg' in model and not clip_guidance_scale else 1
            self.n = self.batch_size
        else:
            self.batch_size = n
            self.n = n
        steps = int(steps)
        self.seed = seed
        self.steps = steps
        self.eta = torch.linspace(0.0, 1.0, 900 + 1)[steps - 100]
        self.outdir = outdir
        self.cutn = cutn
        self.clip_guidance_scale = clip_guidance_scale
        self.transform = transform
        self.init = init
        self.skip_steps = skip_steps
        self.model = model
        self.progress_bar_fn = progress_bar_fn
        print(
            f'NewGenDiffusionDreamer({size=}, {images=}, {seed=}, {steps=}, {self.eta=}, {outdir=}, {cutn=}, {clip_guidance_scale=}, {transform=}, {init=}, {skip_steps=})')

    def type(self):
        return 'diffusion'

    def generate(self, prompt, out_dir):
        sample_fn = sample
        if 'cfg' in self.model:
            sample_fn = cfg_sample

        return sample_fn(prompt, self.images, self.batch_size, self.n, int(self.seed), int(self.steps), self.eta,
                         out_dir,
                         cutn=self.cutn,
                         clip_guidance_scale=self.clip_guidance_scale,
                         apply_transforms=self.transform,
                         size=self.size,
                         model_name=self.model,
                         progress_bar_fn=self.progress_bar_fn,
                         init_url=self.init,
                         skip_steps=self.skip_steps)

    def upsampler(self):
        return latent_upscale

    def same_arguments(self, _):
        return False

    def close(self):
        ...


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--images', type=str, nargs='+', default=[])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('prompt', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    print(args)
    prompt = ' '.join(args.prompt)
    NewGenDiffusionDreamer(args.size, args.images, args.batch_size, args.n, args.seed, args.steps, args.eta,
                           args.outdir).generate([(prompt, 1.0)], args.outdir)
