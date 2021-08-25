import asyncio
import datetime
import os
import shutil
import threading
from typing import Dict, Callable

import clip
import discord
from discord import Thread
from discord.abc import Messageable

from clip_generators.models.taming_transformers.clip_generator.dreamer import load_vqgan_model
from clip_generators.models.taming_transformers.clip_generator.trainer import Trainer
from clip_generators.bots.clip_generator_irc import GenerationArgs, parse_prompt_args
from clip_generators.models.guided_diffusion_hd.clip_guided import Trainer as Diffusion_trainer


class DreamerClient(discord.Client):
    def __init__(self, **options):
        super().__init__(**options)

        self.clip = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
        self.current_user = None
        self.stop_flag = False
        self.commands = self.make_commands()
        self.generating = False
        self.generating_thread = None

    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')

    def make_commands(self) -> Dict[str, Callable[[discord.Message], None]]:
        return {
            '!generate': self.generate_command,
            '!generate_legacy': self.generate_command,
            '!stop': self.stop_command,
            '!leave': self.leave_command,
        }

    def stop_command(self, message: discord.Message):
        if self.current_user == message.author:
            self.stop_flag = True
        else:
            self.loop.create_task(message.channel.send(f'Only {self.current_user} can stop me'))

    def leave_command(self, message: discord.Message):
        self.loop.create_task( message.guild.leave())


    def generate_command(self, message: discord.Message):
        if self.generating:
            self.loop.create_task(message.channel.send(f'already generating for {self.current_user}'))
            return
        prompt = message.content[len("!generate"):]
        if '--prompt' in prompt:
            try:
                arguments = parse_prompt_args(prompt)
            except Exception as ex:
                print(ex)
                self.loop.create_task(message.channel.send('error: ' + str(ex)))
                return
        else:
            arguments = parse_prompt_args('--prompt "osef;1"')
            arguments.prompts = [(prompt, 1.0)]

        print(arguments)
        if arguments.network_type == 'diffusion':
            trainer = self.generate_image_diffusion(arguments)
        else:
            trainer = self.generate_image(arguments)
        self.arguments = arguments

        self.current_user = message.author
        self.stop_flag = False
        self.generating = True

        self.generating_thread = threading.Thread(target=self.train, args=(trainer, message))
        self.generating_thread.start()

    async def on_message(self, message: discord.Message):
        # Thread
        if isinstance(message.channel, Thread):
            return
        if message.author == self.user:
            return

        print(f'{message.channel!r}')

        args = message.content.split()
        if args[0] in self.commands:
            self.commands[args[0]](message)

    async def send_progress(self, trainer, channel, iteration):
        if iteration == trainer.steps:
            await channel.send(trainer.prompt)
        await channel.send(f'step {iteration} / {trainer.steps}', file=discord.File(trainer.get_generated_image_path()))

    async def _train(self, trainer, message: discord.Message):
        if message.guild is not None:
            channel = await message.create_thread(name=trainer.prompt, )
        else:
            channel = message.channel

        await channel.send('generating...')
        now = datetime.datetime.now()
        try:
            for it in trainer.epoch():
                if it % 100 == 0:
                    await self.send_progress(trainer, channel, it)
                await asyncio.sleep(0)
                if self.stop_flag:
                    break

            await channel.send('Done !')
            trainer.close()
            shutil.copy(trainer.get_generated_image_path(),
                        f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.current_user}_{trainer.prompt.replace("//", "_")}.png')
            await self.send_progress(trainer, channel, trainer.steps)
            await message.reply(f'', file=discord.File(trainer.get_generated_image_path()))
        except Exception as ex:
            print(ex)
            await channel.send(str(ex))
        self.generating = False

    def train(self, trainer, message: discord.Message):
        self.loop.create_task(self._train(trainer, message))

    def generate_image_diffusion(self, arguments: GenerationArgs):
        now = datetime.datetime.now()

        trainer = Diffusion_trainer(arguments.prompts,
                                    self.clip,
                                    outdir=f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.current_user}_{arguments.prompts[0][0]}',
                                    )
        return trainer

    def generate_image(self, arguments: GenerationArgs):
        now = datetime.datetime.now()
        trainer = Trainer(arguments.prompts,
                          vqgan_model=load_vqgan_model(arguments.config, arguments.checkpoint).to('cuda'),
                          clip_model=self.clip,
                          learning_rate=arguments.learning_rate,
                          save_every=arguments.refresh_every,
                          outdir=f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.current_user}_{arguments.prompts[0][0]}',
                          device='cuda:0',
                          image_size=(600, 600),
                          crazy_mode=arguments.crazy_mode,
                          cutn=arguments.cut,
                          steps=arguments.steps,
                          full_image_loss=True,
                          nb_augments=1,
                          )
        return trainer


TOKEN = os.getenv('DISCORD_API_KEY')
client = DreamerClient()
client.run(TOKEN)
