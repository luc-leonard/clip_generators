#!/usr/bin/env python3

"""CLIP guided sampling from a diffusion model."""

import argparse
import time
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from clip import clip
from .diffusion import get_model, get_models, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)


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


def sample(size=None, prompts=[], images=[], batch_size=1, n=1, seed=0, steps=100, eta=0, outdir='.'):
    device = 'cuda'
    model = get_model('cc12m_1')()
    _, side_y, side_x = model.shape
    if size is not None:
        side_x = side_y = int(size)

    checkpoint = MODULE_DIR / f'checkpoints/cc12m_1.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model = clip.load(model.clip_model, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    cutn = 16
    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn=cutn, cut_pow=1)

    target_embeds, weights = [], []

    for prompt in prompts:
        txt, weight = prompt
        print(txt, weight)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in images:
        path, weight = prompt
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
        loss = losses.mean(0).sum() * 500
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    def run(x, steps, clip_embed):
        t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        extra_args = {'clip_embed': clip_embed}
        return sampling.cond_sample(model, x, steps, eta, extra_args, cond_fn)

    def run_all(n, batch_size):
        x = torch.randn([n, 3, side_y, side_x], device=device)
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            outs = run(x[i:i + cur_batch_size], steps, clip_embed[i:i + cur_batch_size])
            for j, out in enumerate(outs):
                utils.to_pil_image(out).save(f'./{outdir}/out.png')
                yield f'{outdir}/out.png'

    try:
        return run_all(n, batch_size)
    except KeyboardInterrupt:
        pass


class NewGenDiffusionDreamer():
    def __init__(self, size=None, images=[], batch_size=1, n=1, seed=0, steps=100, eta=0, outdir='.'):
        self.size = size
        self.images = images
        self.batch_size = batch_size
        self.n = n
        self.seed = seed
        self.steps = steps
        self.eta = eta
        self.outdir = outdir
        Path(outdir).mkdir(parents=True, exist_ok=True)

    def type(self):
        return 'diffusion'

    def generate(self, prompt, out_dir):
        return sample(self.size, prompt, self.images, self.batch_size, self.n, int(self.seed), self.steps, self.eta, out_dir)

    def get_generated_image_path(self):
        return f'{self.outdir}/out.png'

    def close(self):
        ...