import datetime
import time

from clip_generators.models.taming_transformers.clip_generator.generator import load_vqgan_model
from clip_generators.utils import GenerationArgs
from clip_generators.models.guided_diffusion_hd.clip_guided_old import Dreamer as Diffusion_dreamer_legacy
from clip_generators.models.guided_diffusion_hd.clip_guided import Dreamer as Diffusion_dreamer
from clip_generators.models.guided_diffusion_hd.clip_guided_new import Dreamer as Diffusion_dreamer_new
from clip_generators.models.taming_transformers.clip_generator.dreamer import Dreamer
from clip_generators.utils import name_filename_fat32_compatible, get_out_dir
from clip_generators.models.new_diffusion.clip_sample import NewGenDiffusionDreamer
from clip_generators.models.glide.dreamer import GlideDreamer
from clip_generators.utils import GlideGenerationArgs


class Generator:
    def __init__(self, args: GenerationArgs, progress_bar_fn, user: str):
        self.args = args
        self.user = user
        self.progress_bar_fn = progress_bar_fn
        if args.network_type == 'diffusion':
            self.dreamer = self.make_dreamer_diffusion(args)
        if args.network_type == 'legacy_diffusion':
            self.dreamer = self.make_dreamer_legacy_diffusion(args)
        elif args.network_type == 'vqgan':
            self.dreamer = self.make_dreamer_vqgan(args)
        elif args.network_type == 'glide':
            self.dreamer = self.make_dreamer_glide(args)

    def make_dreamer_diffusion(self, arguments: GenerationArgs):
        now = datetime.datetime.now()
        return NewGenDiffusionDreamer(arguments.model_arguments.size, [],
                                      progress_bar_fn=self.progress_bar_fn,
                                      seed=arguments.seed,
                                      steps=arguments.steps,
                                      cutn=arguments.cut,
                                      transform=arguments.transforms,
                                      skip_steps=arguments.model_arguments.skips,
                                      n=arguments.model_arguments.n,
                                      init=arguments.resume_from,
                                      model=arguments.model_arguments.model,
                                      eta=0.0 if arguments.model_arguments.ddim_respacing else 1.0,
                                      clip_guidance_scale=arguments.model_arguments.clip_guidance_scale,
                                      outdir=name_filename_fat32_compatible(get_out_dir() / f'{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}'))


    def make_dreamer_vqgan(self, arguments: GenerationArgs):
        now = datetime.datetime.now()
        trainer = Dreamer(arguments.prompts,
                          vqgan_model=load_vqgan_model(arguments.model_arguments.config, arguments.model_arguments.checkpoint).to('cuda'),
                          learning_rate=arguments.model_arguments.learning_rate,
                          save_every=arguments.refresh_every,
                          outdir=name_filename_fat32_compatible(get_out_dir() / f'{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{self.user}_{arguments.prompts[0][0]}'),
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

    def make_dreamer_glide(self, arguments: GenerationArgs):
        model_arguments: GlideGenerationArgs = arguments.model_arguments
        return GlideDreamer(27, model_arguments.clip_guidance_scale / 1000, arguments.steps, model_arguments.upsample_steps)

    def make_dreamer_legacy_diffusion(self, args):
        return Diffusion_dreamer_new(seed=args.seed)

