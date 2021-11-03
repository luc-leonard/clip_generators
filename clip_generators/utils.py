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
    return Path(os.getenv('OUT_DIR', './discord_out_diffusion'))


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
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--cut', type=int, default=40)
    parser.add_argument('--transforms', type=int, default=127)
    parser.add_argument('--full-image-loss', type=bool, default=True)
    parser.add_argument('--network', type=str, default='imagenet')
    parser.add_argument('--network-type', type=str, default='diffusion')
    parser.add_argument('--ddim', dest='ddim_respacing', action='store_true')
    parser.add_argument('--no-ddim', dest='ddim_respacing', action='store_false')
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--init-noise-factor', type=float, default=0.0)
    parser.add_argument('--perlin', type=str, default='')

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
    else:
        return GuidedDiffusionGeneratorArgs(skips=parsed_args.skip,
                                            perlin=parsed_args.perlin,
                                            ddim_respacing=parsed_args.ddim_respacing)

def parse_prompt_args(prompt: str = '') -> GenerationArgs:
    parser = make_arguments_parser()
    try:
        parsed_args = parser.parse_args(shlex.split(prompt))
        print('arguments', parsed_args)
        args = GenerationArgs(prompt=parsed_args.prompt,
                              refresh_every=parsed_args.refresh_every,
                              resume_from=parsed_args.resume_from,
                              steps=parsed_args.steps,
                              cut=parsed_args.cut,
                              network_type=parsed_args.network_type,
                              seed=parsed_args.seed,
                              skips=parsed_args.skip,
                              model_arguments=make_model_arguments(parsed_args)
                              )
        args.prompts = []
        for the_prompt in parsed_args.prompt:
            if ';' in the_prompt:
                separator_index = the_prompt.rindex(';')
                args.prompts.append((the_prompt[:separator_index], float(the_prompt[separator_index + 1:])))
            else:
                args.prompts.append((the_prompt, 1.0))
        print('parsed arguments', args)
        return args
    except SystemExit:
        raise Exception(parser.usage())
