import argparse
import datetime
import gc
import shutil
import threading
import time
import urllib.parse
from pathlib import Path

import PIL.Image
import clip
import irc
import irc.bot
import torch

from clip_generators.bots.miner import Miner
from clip_generators.bots.generator import Generator
from clip_generators.utils import GenerationArgs, parse_prompt_args, get_out_dir, name_filename_fat32_compatible
from clip_generators.models.upscaler.upscaler import upscale
from clip_generators.models.rudalle.rudalle import RudalleGenerator


class IrcBot(irc.bot.SingleServerIRCBot):

    def __init__(self, channel: str, nickname: str, server: str, port=6667):
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port)], nickname, nickname)
        self.generating = None
        self.generating_thread = None
        self.channel = channel
        self.current_generating_user = None
        # self.stop_generating = False
        # print('loading clip')
        # self.clip = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
        #self.miner = Miner('/home/lleonard/t-rex/launch.sh')
       # self.miner.start()
        self.clip = None
        self.dalle_generator = None

    def on_nicknameinuse(self, c: irc.client, e):
        c.nick(c.get_nickname() + "_")

    def on_welcome(self, c, e):
        print('connected to server')
        c.join(self.channel)


    def on_privmsg(self, c: irc.client.ServerConnection, e: irc.client.Event):
        ...



    def train(self, trainer, args: GenerationArgs, c):
        now = datetime.datetime.now()
        out_dir = name_filename_fat32_compatible(
            get_out_dir() / f'{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.current_generating_user.nick}_{args.prompts[0][0]}')

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / 'args.txt').write_text(args.json())
        for it in trainer.generate(args.prompts, out_dir):
            if self.stop_generating is True:
                break
            if it is not None:
                if it[0] == 0:
                    c.privmsg(self.channel, f'{it[0]} => http://home.luc-leonard.fr:8082/{Path(it[2]).relative_to(get_out_dir())}')
                else:
                    last_it = it
                    c.privmsg(self.channel, f'{it[0]}')

        trainer.close()

        last_path = name_filename_fat32_compatible(out_dir.parent / f'{now.isoformat()}_{args.prompts[0][0]}.png')
        shutil.copyfile(last_it[1], str(last_path))

        if args.upsample and hasattr(trainer, 'upsampler'):
            hd_path = name_filename_fat32_compatible(
                get_out_dir() / f'{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.current_generating_user.nick}_{args.prompts[0][0]}_hd.png')
            half_path = name_filename_fat32_compatible(
                get_out_dir() / f'{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.current_generating_user.nick}_{args.prompts[0][0]}_half.png')
            trainer.upsampler()(last_path, hd_path)
            PIL.Image.open(hd_path).resize((1024, 1024)).save(half_path)
            c.privmsg(self.channel, f'http://home.luc-leonard.fr:8082/{Path(hd_path).relative_to(get_out_dir())}')

        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        # self.miner.start()
        self.generating = None
        c.privmsg(self.channel, f'Generation done. Ready to take an order')


    def on_pubmsg(self, c, e):
        text = e.arguments[0]
        args = text.split()

        if args[0] == '!generate' or args[0] == '!g':
            #self.miner.stop()
            if self.generating is not None:
                c.privmsg(e.target, f'currently generating {self.generating}, try again later.')
                return
            prompt = ' '.join(args[1:])
            try:
                arguments = parse_prompt_args(prompt)
            except Exception as ex:
                print(ex)
                c.privmsg(e.target, str(ex))
                return
            c.privmsg(e.target, f'generating {arguments.prompts[0]}')
            trainer = Generator(arguments, self.clip, e.source.nick).dreamer


            self.generating = prompt
            self.stop_generating = False
            self.current_generating_user = e.source
            self.generating_thread = threading.Thread(target=self.train, args=(trainer, arguments, c))
            self.generating_thread.start()

        if args[0] == '!' + 'stop' and e.source == self.current_generating_user:
            self.stop_generating = True
            if self.dalle_generator:
                self.dalle_generator.stop()

        if args[0] == '!' + 'd' and self.generating is None:
            self.current_generating_user = e.source
            self.generating_thread = threading.Thread(target=self.dalle, args=(c, e))
            self.generating_thread.start()
        if args[0] == '!' + 'restart':
            self.dalle_generator = None

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
    try:
        main()
    except Exception as ex:
        print(ex)
