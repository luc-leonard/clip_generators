import itertools
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from progressbar import progressbar
from torch import optim
from torchvision.transforms import functional as TF

from .discriminator import ClipDiscriminator
from .dreamer import Generator
from .dreamer import ZSpace

current_path = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(current_path / '..'))


def network_list():
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    models_path = current_path / '..' / 'models'
    return {
        'ffhq': {
            'config': models_path / 'ffhq' / 'configs' / '2020-11-13T21-41-45-project.yaml',
            'checkpoint': models_path / 'ffhq' / 'checkpoints' / 'last.ckpt'
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


class Trainer:
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
                 save_every=50):

        if seed is None:
            torch.manual_seed(int(time.time()))
        else:
            torch.manual_seed(seed)
        self.save_every = save_every
        self.outdir = Path(outdir)
        if steps is None:
            self.iterator = itertools.count(start=0)
        else:
            self.iterator = range(steps)
        self.steps = steps
        self.prompts = prompts
        self.prompt = prompts[0]
        self.outdir.mkdir(exist_ok=True, parents=True)
        (self.outdir / 'prompt.txt').write_text('\n'.join(prompts))
        self.clip_discriminator = ClipDiscriminator(clip_model, prompts, cutn, cut_pow, device,
                                                    full_image_loss=full_image_loss,
                                                    nb_augments=nb_augments )

        self.generator = Generator(vqgan_model).to(device)
        self.z_space = ZSpace(vqgan_model, image_size, device=device)
        self.optimizer = optim.Adam([self.z_space.z], lr=learning_rate)
        self.scheduler = None
        if crazy_mode is True and steps is not None:
           self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate * 10, total_steps=steps)
        self.video = imageio.get_writer(f'{outdir}/out.mp4', mode='I', fps=5, codec='libx264', bitrate='16M')

    def get_generated_image_path(self):
        return self.outdir / f'progress_latest.png'

    @torch.no_grad()
    def save_image(self, i, generated_image, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        pil_image = TF.to_pil_image(generated_image[0].cpu())
        self.video.append_data(np.array(pil_image))
        pil_image.save(str(self.outdir / f'progress_latest.png'))

    def start(self):
        for _ in self.epoch():
            ...

    def close(self):
        self.video.close()

    def epoch(self):
        for i in progressbar(self.iterator):
            self.optimizer.zero_grad()
            generated_image = self.generator(self.z_space.z)
            losses = self.clip_discriminator(generated_image)
            if i % self.save_every == 0:
                self.save_image(i, generated_image, losses)
            yield i

            sum(losses).backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.z_space.clamp()
            i = i + 1
