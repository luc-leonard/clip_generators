import argparse
import io
import shlex
import time
from pathlib import Path
from typing import Optional, Tuple, List

import requests
from pydantic import BaseModel

from clip_generators.models.taming_transformers.clip_generator.dreamer import network_list


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


class GenerationArgs(BaseModel):
    steps: int = 500
    network_type: str
    refresh_every: int = 100
    resume_from: Optional[str] = None
    network: str = 'imagenet',
    cut: int = 64
    nb_augments: int = 3
    full_image_loss: bool = True
    prompts: List[Tuple[str, float]] = []
    crazy_mode: bool = False
    learning_rate: float = 0.05
    ddim_respacing: bool = False
    seed: int
    skips: int

    @property
    def config(self):
        return Path('..') / networks[self.network]['config']

    @property
    def checkpoint(self):
        return Path('..') / networks[self.network]['checkpoint']


def make_arguments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--crazy-mode', type=bool, default=False)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--refresh-every', type=int, default=10)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--cut', type=int, default=64)
    parser.add_argument('--transforms', type=int, default=1)
    parser.add_argument('--full-image-loss', type=bool, default=True)
    parser.add_argument('--network', type=str, default='imagenet')
    parser.add_argument('--network-type', type=str, default='diffusion')
    parser.add_argument('--ddim', dest='ddim_respacing', action='store_true')
    parser.add_argument('--no-ddim', dest='ddim_respacing', action='store_false')
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--skip', type=int, default=0)

    parser.set_defaults(ddim_respacing=False)
    return parser

def parse_prompt_args(prompt: str = '') -> GenerationArgs:
    parser = make_arguments_parser()
    try:
        parsed_args = parser.parse_args(shlex.split(prompt))
        args = GenerationArgs(prompt=parsed_args.prompt,
                              crazy_mode=parsed_args.crazy_mode,
                              learning_rate=parsed_args.learning_rate,
                              refresh_every=parsed_args.refresh_every,
                              resume_from=parsed_args.resume_from,
                              steps=parsed_args.steps,
                              cut=parsed_args.cut,
                              network=parsed_args.network,
                              nb_augments=parsed_args.transforms,
                              full_image_loss=parsed_args.full_image_loss,
                              network_type=parsed_args.network_type,
                              ddim_respacing=parsed_args.ddim_respacing,
                              seed=parsed_args.seed,
                              skips=parsed_args.skip,
                              )
        args.prompts = []
        for the_prompt in parsed_args.prompt:
            if ';' in the_prompt:
                separator_index = the_prompt.rindex(';')
                args.prompts.append((the_prompt[:separator_index], float(the_prompt[separator_index + 1:])))
            else:
                args.prompts.append((the_prompt, 1.0))
        return args
    except SystemExit:
        raise Exception(parser.usage())