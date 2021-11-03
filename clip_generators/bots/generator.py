import datetime

from clip_generators.models.taming_transformers.clip_generator.generator import load_vqgan_model
from clip_generators.utils import GenerationArgs
from clip_generators.models.guided_diffusion_hd.clip_guided_old import Dreamer as Diffusion_dreamer_legacy
from clip_generators.models.guided_diffusion_hd.clip_guided import Dreamer as Diffusion_dreamer
from clip_generators.models.guided_diffusion_hd.clip_guided_new import Dreamer as Diffusion_dreamer_new
from clip_generators.models.taming_transformers.clip_generator.dreamer import Dreamer
from clip_generators.utils import name_filename_fat32_compatible, get_out_dir


class Generator:
    def __init__(self, args: GenerationArgs, clip, user: str):
        self.args = args
        self.clip = clip
        self.user = user
        if args.network_type == 'diffusion':
            self.dreamer = self.make_dreamer_diffusion(args)
        if args.network_type == 'diffusion_2':
            self.dreamer = self.make_dreamer_diffusion_2(args)
        elif args.network_type == 'vqgan':
            self.dreamer = self.make_dreamer_vqgan(args)
        elif args.network_type == 'diffusion_old':
            self.dreamer = self.make_dreamer_diffusion_old(args)

    def make_dreamer_diffusion(self, arguments: GenerationArgs):
        now = datetime.datetime.now()

        trainer = Diffusion_dreamer(arguments.prompts,
                                    self.clip,
                                    init_image=arguments.resume_from,
                                    ddim_respacing=arguments.model_arguments.ddim_respacing,
                                    seed=arguments.seed,
                                    steps=arguments.steps,
                                    outdir=name_filename_fat32_compatible(get_out_dir() / f'{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}'),
                                    skip_timesteps=arguments.model_arguments.skips,
                                    cut=arguments.cut,
                                    cut_batch=arguments.nb_augment,
                                    perlin=arguments.model_arguments.perlin,
                                    )
        return trainer

    def make_dreamer_vqgan(self, arguments: GenerationArgs):
        now = datetime.datetime.now()
        trainer = Dreamer(arguments.prompts,
                          vqgan_model=load_vqgan_model(arguments.model_arguments.config, arguments.model_arguments.checkpoint).to('cuda'),
                          clip_model=self.clip,
                          learning_rate=arguments.model_arguments.learning_rate,
                          save_every=arguments.refresh_every,
                          outdir=name_filename_fat32_compatible(f'/media/lleonard/My Passport/generated_art/out/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}'),
                          device='cuda:0',
                          image_size=(700, 700),
                          crazy_mode=arguments.model_arguments.crazy_mode,
                          cutn=arguments.cut,
                          steps=arguments.steps,
                          full_image_loss=True,
                          nb_augments=1,
                          init_image=arguments.resume_from,
                          init_noise_factor=arguments.model_arguments.init_noise_factor
                          )
        return trainer

    def make_dreamer_diffusion_old(self, arguments):
        now = datetime.datetime.now()

        trainer = Diffusion_dreamer_legacy(arguments.prompts,
                                    self.clip,
                                    init_image=arguments.resume_from,
                                    ddim_respacing=arguments.model_arguments.ddim_respacing,
                                    seed=arguments.seed,
                                    steps=arguments.steps,
                                    outdir=f'/media/lleonard/My Passport/generated_art/out/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}',
                                    skip_timesteps=arguments.model_arguments.skips,
                                    )
        return trainer

    def make_dreamer_diffusion_2(self, arguments):
        now = datetime.datetime.now()

        trainer = Diffusion_dreamer_new(arguments.prompts,
                                    self.clip,
                                    init_image=arguments.resume_from,
                                    ddim_respacing=arguments.model_arguments.ddim_respacing,
                                    seed=arguments.seed,
                                    steps=arguments.steps,
                                    outdir=f'/media/lleonard/My Passport/generated_art/out/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}',
                                    skip_timesteps=arguments.model_arguments.skips,
                                    )
        return trainer

