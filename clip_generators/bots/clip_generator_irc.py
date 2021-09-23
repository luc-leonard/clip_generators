import argparse
import datetime
import shutil
import threading
import urllib.parse

import clip
import irc
import irc.bot

from clip_generators.bots.miner import Miner
from clip_generators.bots.generator import Generator
from clip_generators.models.guided_diffusion_hd.clip_guided import Dreamer as Diffusion_trainer
from clip_generators.models.taming_transformers.clip_generator.generator import load_vqgan_model
from clip_generators.models.taming_transformers.clip_generator.dreamer import Dreamer
from clip_generators.utils import GenerationArgs, parse_prompt_args


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
        self.miner = Miner('/home/lleonard/t-rex/launch.sh')
        self.miner.start()

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
        self.miner.start()
        self.generating = None
        c.privmsg(self.channel, f'Generation done. Ready to take an order')
        shutil.copy(trainer.get_generated_image_path(),
                    f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{trainer.prompts[0][0].replace("//", "_")}.png')

    def on_pubmsg(self, c, e):
        text = e.arguments[0]
        args = text.split()

        if args[0] == '!' + 'generate':
            self.miner.stop()
            if self.generating is not None:
                c.privmsg(e.target, f'currently generating {self.generating}, try again later.')
                return
            prompt = ' '.join(args[1:])
            try:
                arguments = parse_prompt_args(prompt)
            except Exception as ex:
                c.privmsg(e.target, str(ex))
                return
            c.privmsg(e.target, f'generating {arguments.prompts[0]}')
            trainer = Generator(arguments, self.clip, '__IRC__').dreamer
            generated_image_path = trainer.get_generated_image_path()
            c.privmsg(e.target,
                      f'{prompt} => http://home.luc-leonard.fr:8082/{urllib.parse.quote(str(generated_image_path.relative_to(".")))}')


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
