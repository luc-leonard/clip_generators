import itertools
import json
import os
import sys
import time
import urllib
from pathlib import Path

import PIL
import imageio
import numpy as np
import torch
from progressbar import progressbar
from torch import optim
from torchvision.transforms import functional as TF

from .discriminator import ClipDiscriminator
from .generator import Generator
from .generator import ZSpace

current_path = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(current_path / '..'))


def network_list():
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    models_path = current_path / '..' / 'models'
    return {
        'ffhq': {
            'config': models_path / 'ffhq' / 'configs' / '2020-11-13T21-41-45-project.yaml',
            'checkpoint': models_path / 'ffhq' / 'checkpoints' / 'last_1.ckpt'
        },
        'imagenet': {
            'config': models_path / 'imagenet' / 'vqgan_imagenet_f16_16384.yaml',
            'checkpoint': models_path / 'imagenet' / 'vqgan_imagenet_f16_16384.ckpt'
        },
        'wikiart': {
            'config': models_path / 'wikiart' / 'configs' / 'wikiart_f16_16384_8145600.yaml',
            'checkpoint': models_path / 'wikiart' / 'checkpoints' / 'wikiart_f16_16384_8145600.ckpt',
        },
        'coco': {
            'config': models_path / 'coco' / 'coco.yaml',
            'checkpoint': models_path / 'coco' / 'coco.ckpt'
        }
    }


class Dreamer:
    def __init__(self,
                 prompts,
                 vqgan_model,
                 clip_model,
                 device='cuda:0',
                 learning_rate=0.05,
                 outdir='./out',
                 image_size=(512, 512),
                 cutn=64,
                 cut_pow=1.,
                 seed=None,
                 steps=None,
                 crazy_mode=False,
                 nb_augments=3,
                 full_image_loss=True,
                 save_every=10,
                 init_image=None,
                 init_noise_factor=0.):

        if seed is None:
            seed = int(time.time())
        torch.manual_seed(seed)

        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

        self.save_every = save_every


        if steps is None:
            self.iterator = itertools.count(start=0)
        else:
            self.iterator = range(steps)
        self.steps = steps
        self.prompts = prompts
        self.prompt = prompts[0][0]

        self.clip_discriminator = ClipDiscriminator(clip_model, prompts, cutn, cut_pow, device,
                                                    full_image_loss=full_image_loss,
                                                    nb_augments=nb_augments)

        self.generator = Generator(vqgan_model).to(device)
        self.z_space = ZSpace(self.generator, image_size, device=device, init_image=init_image, init_noise_factor=init_noise_factor)
        if init_image is not None:
            self.z_space.base_image.save(str(self.outdir / 'base.png'))
            TF.to_pil_image(self.z_space.base_image_decoded[0].cpu()).save(str(self.outdir / 'projection.png'))
        self.optimizer = optim.Adam([self.z_space.z], lr=learning_rate)
        self.scheduler = None
        if crazy_mode is True and steps is not None:
           self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate * 10, total_steps=steps)
        self.video = imageio.get_writer(f'{outdir}/out.mp4', mode='I', fps=25, codec='libx264', bitrate='16M')

    def get_generated_image_path(self):
        return self.outdir / f'progress_latest.png'

    @torch.no_grad()
    def save_image(self, i, generated_image, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        pil_image = TF.to_pil_image(generated_image[0].cpu())
        pil_image.save(str(self.outdir / f'progress_latest.png'))
        return pil_image

    def start(self):
        for _ in self.epoch():
            ...

    def close(self):
        self.video.close()

    def epoch(self):
        loss = torch.nn.MSELoss()
        for i in progressbar(self.iterator):
            self.optimizer.zero_grad()
            generated_image = self.generator(self.z_space.z)
            losses = self.clip_discriminator(generated_image)
            if self.z_space.base_image is not None and self.clip_discriminator.full_image_loss:
                losses.append(loss(generated_image, self.z_space.base_image_decoded) * 0.5)
            if i % self.save_every == 0:
                image = self.save_image(i, generated_image, losses)
                self.video.append_data(np.array(image))
            yield i

            sum(losses).backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.z_space.clamp()
            i = i + 1
