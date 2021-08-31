import datetime
import argparse
import datetime
import shlex
import shutil
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import clip
import irc
import irc.bot
from pydantic import BaseModel

from clip_generators.models.taming_transformers.clip_generator.dreamer import load_vqgan_model
from clip_generators.models.taming_transformers.clip_generator.trainer import Trainer, network_list
from clip_generators.models.guided_diffusion_hd.clip_guided import Trainer as Diffusion_trainer
from clip_generators.models.taming_transformers.clip_generator.discriminator import EmbeddedText

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
    ddim_respacing: bool = True

    @property
    def config(self):
        return Path('..') / networks[self.network]['config']

    @property
    def checkpoint(self):
        return Path('..') / networks[self.network]['checkpoint']


def parse_prompt_args(prompt: str = '') -> GenerationArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument('--crazy-mode', type=bool, default=False)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--refresh-every', type=int, default=10)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--cut', type=int, default=64)
    parser.add_argument('--transforms', type=int, default=1)
    parser.add_argument('--full-image-loss', type=bool, default=True)
    parser.add_argument('--network', type=str, default='imagenet')
    parser.add_argument('--network-type', type=str, default='diffusion')
    parser.add_argument('--ddim-respacing', type=bool, default=True)
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
                              ddim_respacing=parsed_args.ddim_respacing
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


class IrcBot(irc.bot.SingleServerIRCBot):

    def __init__(self, channel: str, nickname: str, server: str, port=6667):
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port)], nickname, nickname)
        self.generating = None
        self.generating_thread = None
        self.channel = channel
        self.current_generating_user = None
        self.stop_generating = False
        print('loading clip')
        self.clip = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')

    def on_nicknameinuse(self, c: irc.client, e):
        c.nick(c.get_nickname() + "_")

    def on_welcome(self, c, e):
        c.join(self.channel)

    def on_privmsg(self, c: irc.client.ServerConnection, e: irc.client.Event):
        ...

    def train(self, trainer, c):
        now = datetime.datetime.now()
        for it in trainer.epoch():
            if self.stop_generating is True:
                break
            if it % 100 == 0:
                c.privmsg(self.channel, f'generation {it}/{trainer.steps}')
        trainer.close()
        self.generating = None
        c.privmsg(self.channel, f'Generation done. Ready to take an order')
        shutil.copy(trainer.get_generated_image_path(),
                    f'./irc_out/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{trainer.prompts[0].replace("//", "_")}.png')

    def generate_image(self, arguments: GenerationArgs):
        vqgan_model = load_vqgan_model(arguments.config, arguments.checkpoint).to('cuda')
        now = datetime.datetime.now()
        trainer = Trainer([arguments.prompt],
                          vqgan_model,
                          self.clip,
                          learning_rate=arguments.learning_rate,
                          save_every=arguments.refresh_every,
                          outdir=f'./irc_out/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{arguments.prompt}',
                          device='cuda:0',
                          image_size=(640, 640),
                          crazy_mode=arguments.crazy_mode,
                          cutn=arguments.cut,
                          steps=arguments.steps,
                          full_image_loss=arguments.full_image_loss,
                          nb_augments=arguments.nb_augments,
                          )
        return trainer

    def generate_image_diffusion(self, arguments: GenerationArgs):
        print(arguments)
        now = datetime.datetime.now()
        trainer = Diffusion_trainer(arguments.prompt.split('||')[0], self.clip,
                                    outdir=f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{arguments.prompt}',
                                    init_image=arguments.resume_from,
                                    )
        return trainer

    def on_pubmsg(self, c, e):
        text = e.arguments[0]
        args = text.split()

        if args[0] == '!' + 'generate':
            if self.generating is not None:
                c.privmsg(e.target, f'currently generating {self.generating}, try again later.')
                return
            prompt = ' '.join(args[1:])
            try:
                arguments = parse_prompt_args(prompt)
            except Exception as ex:
                c.privmsg(e.target, str(ex))
                return
            c.privmsg(e.target, f'generating {arguments.prompt}')
            trainer = self.generate_image_diffusion(arguments)
            generated_image_path = trainer.get_generated_image_path()
            c.privmsg(e.target,
                      f'{prompt} => http://82.65.144.151:8082/{urllib.parse.quote(str(generated_image_path.relative_to(".")))}')
            self.generating = prompt
            self.stop_generating = False
            self.current_generating_user = e.source
            self.generating_thread = threading.Thread(target=self.train, args=(trainer, c))
            self.generating_thread.start()

        if args[0] == '!' + 'stop' and e.source == self.current_generating_user:
            self.stop_generating = True

    def on_dccmsg(self, c, e):
        pass

    def on_dccchat(self, c, e):
        pass

    def do_command(self, e, cmd):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", help="the name")

    parser.add_argument("--server", help="the model")
    parser.add_argument("--channel", help="the model")

    return parser.parse_args()


def main():
    args = parse_arguments()
    bot = IrcBot(args.channel, args.name, args.server)

    bot.start()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
