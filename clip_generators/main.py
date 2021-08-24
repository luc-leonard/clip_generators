import sys


from clip_generators.models.guided_diffusion_hd.clip_guided import Trainer as Diffusion_trainer
from clip_generators.models.taming_transformers.clip_generator.trainer import Trainer
import clip

from clip_generators.models.taming_transformers.clip_generator.dreamer import load_vqgan_model
from clip_generators.models.taming_transformers.clip_generator.trainer import download_models, network_list


def generate(prompt):
    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
    model_path = network_list()['imagenet']
    trainer = Trainer([prompt], vqgan_model=load_vqgan_model(model_path['config'], model_path['checkpoint']).to('cuda'),
                      clip_model=clip_model, outdir=f'./out/{prompt}/')
    for it in trainer.epoch():
        ...


if __name__ == '__main__':
    download_models()
    generate(sys.argv[-1])
