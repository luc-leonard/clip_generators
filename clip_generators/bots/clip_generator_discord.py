import asyncio
import datetime
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Callable

import PIL
import clip
import discord
import progressbar
import torch
from discord import Thread
from discord.abc import Messageable

from clip_generators.bots.generator import Generator
from clip_generators.bots.miner import Miner
from clip_generators.utils import GenerationArgs, parse_prompt_args
from clip_generators.utils import make_arguments_parser
from clip_generators.models.upscaler.upscaler import upscale
from clip_generators.models.taming_transformers.clip_generator.generator import fetch
from clip_generators.utils import name_filename_fat32_compatible, get_out_dir
from clip_generators.models.rudalle.rudalle import RudalleGenerator
from rudalle import get_realesrgan


import warnings
warnings.filterwarnings("ignore")

class DreamerClient(discord.Client):
    def __init__(self, **options):
        super().__init__(**options)

        self.clip = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
        self.current_user = None
        self.stop_flag = False
        self.commands = self.make_commands()

        self.generating = False
        self.generating_thread = None
        self.defaults = {}

        self.miner_enabled = True
        self.miner = Miner('/home/lleonard/t-rex/launch.sh')
        self.current_dreamer = None

    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')

    def make_commands(self) -> Dict[str, Callable[[discord.Message], None]]:
        return {
            '!generate': self.generate_command,
            '!g': self.generate_command,
            '!set': self.set_command,
            '!stop': self.stop_command,

           # '!leave': self.leave_command,
            '!help': self.help_command,
            '!mine': self.mine_command,
            '!upscale': self.upscale_command,
            '!restart': self.restart_command
        }

    async def on_message(self, message: discord.Message):
        if isinstance(message.channel, Thread):
            return
        if message.author == self.user:
            return

        print(message.content)
        args = message.content.split()
        if args[0] in self.commands:
            self.commands[args[0]](message)


    def set_command(self, message):
        ...

    def upscale_command(self, message):
        remote_url = message.content[len('!upscale') + 1:]
        with fetch(remote_url) as remote_file:
            with open('/tmp/temp', 'wb') as local_file:
                local_file.write(remote_file.read())
                upscale('/tmp/temp', '/tmp/temp_hd.png')
                hd_image = PIL.Image.open('/tmp/temp_hd.png')
                hd_image.thumbnail((1024, 1024))
                stream = io.BytesIO()

                hd_image.save(stream, format='PNG')
                stream.seek(0)
                self.loop.create_task(message.reply(f'', files=[discord.File(stream, filename='upsampled.png')]))


    def restart_command(self, message):
        self.dalle_generator = None
        # self.
        torch.cuda.empty_cache()
        gc.collect()
        self.loop.create_task(message.channel.send('Restarted'))

    def mine_command(self, message):
        prompt = message.content[len("!mine") + 1:]
        if prompt == 'enabled':
            self.miner.start()
        elif prompt == 'disabled':
            self.miner.stop()

    def help_command(self, message: discord.Message):
        self.loop.create_task(message.channel.send(make_arguments_parser().usage()))

    def stop_command(self, message: discord.Message):
        if self.current_user == message.author:
            self.stop_flag = True
        else:
            self.loop.create_task(message.channel.send(f'Only {self.current_user} can stop me'))

    def leave_command(self, message: discord.Message):
        self.loop.create_task(message.guild.leave())

    def generate_command(self, message: discord.Message):
        self.dalle_generator = None
        torch.cuda.empty_cache()
        gc.collect()

        if self.generating:
            self.loop.create_task(message.channel.send(f'already generating for {self.current_user}'))
            return

        self.miner.stop()

        torch.cuda.empty_cache()
        gc.collect()

        prompt = message.content[message.content.index(' ') + 1:]
        try:
            arguments = parse_prompt_args(prompt, 'glide')
        except Exception as ex:
            print(ex)
            self.loop.create_task(message.channel.send('error: ' + str(ex)))
            return

        self.current_user = message.author

        if self.current_dreamer is None or arguments.network_type != self.current_dreamer.type() or self.current_dreamer.same_arguments(arguments) is False:
            dreamer = Generator(arguments, self.clip, str(self.current_user)).dreamer
            self.current_dreamer = dreamer
        else:
            dreamer = self.current_dreamer

        self.arguments = arguments
        self.stop_flag = False
        self.generating = True

        self.loop.create_task(self.generate(dreamer, message, arguments))

    async def send_progress(self, channel, iteration):
        await channel.send(f'step {iteration[0]}', file=discord.File(iteration[1]))

    async def generate(self, dreamer, message: discord.Message, arguments):
        if message.guild is not None:
            message_to_start_thread = await message.channel.send(arguments.prompts[0][0])
            channel = await message_to_start_thread.create_thread(name=arguments.prompts[0][0][:90], )
        else:
            channel = message.channel
        self.loop.create_task(channel.send('arguments = ' + str(arguments)))
        await channel.send('generating...')

        now = datetime.datetime.now()
        out_dir = name_filename_fat32_compatible(get_out_dir() / f'{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.current_user}_{arguments.prompts[0][0]}')
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / 'args.txt').write_text(arguments.json())
            torch.manual_seed(arguments.seed)
            for it in dreamer.generate(arguments.prompts, out_dir):
                if it is not None:
                    await self.send_progress(channel, it)
                await asyncio.sleep(0)
                if self.stop_flag:
                    break
            dreamer.close()

            await message.reply(f'', files=[discord.File(it[1], filename='upsampled.png')])

        except Exception as ex:
            print(ex)
            print(traceback.format_exc())
            await channel.send(str(ex))

        self.generating = False
        self.miner.start()


TOKEN = os.getenv('DISCORD_API_KEY')
client = DreamerClient()
client.run(TOKEN)
