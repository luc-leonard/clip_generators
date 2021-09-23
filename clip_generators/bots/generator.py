import datetime

from clip_generators.models.taming_transformers.clip_generator.generator import load_vqgan_model
from clip_generators.utils import GenerationArgs
from clip_generators.models.guided_diffusion_hd.clip_guided import Dreamer as Diffusion_dreamer
from clip_generators.models.taming_transformers.clip_generator.dreamer import Dreamer

class Generator:
    def __init__(self, args: GenerationArgs, clip, user: str):
        self.args = args
        self.clip = clip
        self.user = user
        if args.network_type == 'diffusion':
            self.dreamer = self.make_dreamer_diffusion(args)
        else:
            self.dreamer = self.make_dreamer_vqgan(args)

    def make_dreamer_diffusion(self, arguments: GenerationArgs):
        now = datetime.datetime.now()

        trainer = Diffusion_dreamer(arguments.prompts,
                                    self.clip,
                                    init_image=arguments.resume_from,
                                    ddim_respacing=arguments.ddim_respacing,
                                    seed=arguments.seed,
                                    steps=arguments.steps,
                                    outdir=f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}',
                                    skip_timesteps=arguments.skips,
                                    )
        return trainer

    def make_dreamer_vqgan(self, arguments: GenerationArgs):
        now = datetime.datetime.now()
        trainer = Dreamer(arguments.prompts,
                          vqgan_model=load_vqgan_model(arguments.config, arguments.checkpoint).to('cuda'),
                          clip_model=self.clip,
                          learning_rate=arguments.learning_rate,
                          save_every=arguments.refresh_every,
                          outdir=f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}',
                          device='cuda:0',
                          image_size=(700, 700),
                          crazy_mode=arguments.crazy_mode,
                          cutn=arguments.cut,
                          steps=arguments.steps,
                          full_image_loss=True,
                          nb_augments=1,
                          init_image=arguments.resume_from,
                          init_noise_factor=arguments.init_noise_factor
                          )
        return trainer
