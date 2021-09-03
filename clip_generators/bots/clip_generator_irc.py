import argparse
import datetime
import shutil
import threading
import urllib.parse

import clip
import irc
import irc.bot

from clip_generators.models.guided_diffusion_hd.clip_guided import Trainer as Diffusion_trainer
from clip_generators.models.taming_transformers.clip_generator.dreamer import load_vqgan_model
from clip_generators.models.taming_transformers.clip_generator.trainer import Trainer
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
