import argparse
import io
import os
import re
import shlex
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union, Any

import requests
from pydantic import BaseModel

from clip_generators.models.taming_transformers.clip_generator.dreamer import network_list


def get_out_dir() -> Path:
    return Path(os.getenv('OUT_DIR', './res/discord_out_diffusion'))


def name_filename_fat32_compatible(path: Path)->Path:
    return path


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


networks = network_list()


class GuidedDiffusionGeneratorArgs(BaseModel):
    skips: int = 0
    ddim_respacing: bool = True
    perlin: str = None
    size: Optional[int] = None


class VQGANGenerationArgs(BaseModel):
    network: str = 'imagenet'

    full_image_loss: bool = True
    crazy_mode: bool = False
    learning_rate: float = 0.05
    init_noise_factor: float = 0.0

    @property
    def config(self):
        return Path('..') / networks[self.network]['config']

    @property
    def checkpoint(self):
        return Path('..') / networks[self.network]['checkpoint']


class RudalleGenerationArgs(BaseModel):
    nb_images: int = 3
    image_cut_top: Optional[int] = None
    emoji: bool = False


class GlideGenerationArgs(BaseModel):
    ...


class GenerationArgs(BaseModel):
    prompts: List[Tuple[str, float]] = []
    steps: int = 500
    network_type: str
    refresh_every: int = 100
    seed: int
    resume_from: Optional[str] = None
    cut: int = 64
    nb_augment: int = 2
    model_arguments: Any


def make_arguments_parser(**kwargs) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--crazy-mode', type=bool, default=False)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--refresh-every', type=int, default=10)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('prompt', nargs=argparse.REMAINDER)
    parser.add_argument('--cut', type=int, default=40)
    parser.add_argument('--transforms', type=int, default=127)
    parser.add_argument('--full-image-loss', type=bool, default=True)
    parser.add_argument('--network', type=str, default='imagenet')
    parser.add_argument('--network-type', type=str, default='glide')
    parser.add_argument('--ddim', dest='ddim_respacing', action='store_true')
    parser.add_argument('--no-ddim', dest='ddim_respacing', action='store_false')
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--init-noise-factor', type=float, default=0.0)
    parser.add_argument('--perlin', type=str, default='')

    parser.add_argument('--size', default=None, type=int)
    #rudalle
    parser.add_argument('--emoji', default=False, action='store_true')
    parser.add_argument('--images', type=int, default=3)
    parser.add_argument('--cut-top', type=int, default=4)

    parser.set_defaults(ddim_respacing=True)
    parser.set_defaults(**kwargs)
    return parser


def make_model_arguments(parsed_args):
    if parsed_args.network_type == 'vqgan':
        return VQGANGenerationArgs(network=parsed_args.network,
                            nb_augments=parsed_args.transforms,
                            full_image_loss=parsed_args.full_image_loss,
                            crazy_mode=parsed_args.crazy_mode,
                            learning_rate=parsed_args.learning_rate,
                            init_noise_factor=parsed_args.init_noise_factor)
    elif parsed_args.network_type == 'diffusion':
        return GuidedDiffusionGeneratorArgs(skips=parsed_args.skip,
                                            perlin=parsed_args.perlin,
                                            size=parsed_args.size,
                                            ddim_respacing=parsed_args.ddim_respacing)
    elif parsed_args.network_type == 'rudalle':
        return RudalleGenerationArgs(nb_images=parsed_args.images, emoji=parsed_args.emoji, image_cut_top=parsed_args.cut_top)
    elif parsed_args.network_type == 'glide':
        return GlideGenerationArgs()


def parse_prompt_args(prompt: str = '', default_generator='') -> GenerationArgs:
    parser = make_arguments_parser(network_type=default_generator)
    try:
        parsed_args = parser.parse_args(shlex.split(prompt))
        print('arguments', parsed_args)
        args = GenerationArgs(prompts=[(' '.join(parsed_args.prompt), 1.0)],
                              refresh_every=parsed_args.refresh_every,
                              resume_from=parsed_args.resume_from,
                              steps=parsed_args.steps,
                              cut=parsed_args.cut,
                              network_type=parsed_args.network_type,
                              seed=parsed_args.seed,
                              skips=parsed_args.skip,
                              model_arguments=make_model_arguments(parsed_args)
                              )
        print('parsed arguments', args)
        return args
    except SystemExit:
        raise Exception(parser.usage())