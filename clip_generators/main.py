import sys


from clip_generators.models.guided_diffusion_hd.clip_guided import Dreamer as Diffusion_trainer
from clip_generators.models.taming_transformers.clip_generator.dreamer import Dreamer
import clip

from clip_generators.models.taming_transformers.clip_generator.generator import load_vqgan_model
from clip_generators.models.taming_transformers.clip_generator.dreamer import network_list


def generate(prompt):
    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
    model_path = network_list()['imagenet']
    trainer = Dreamer([prompt], vqgan_model=load_vqgan_model(model_path['config'], model_path['checkpoint']).to('cuda'),
                      clip_model=clip_model, outdir=f'./out/{prompt}/')
    for it in trainer.epoch():
        ...


def generate_diffusion(prompt):
    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
    trainer = Diffusion_trainer(prompt, clip_model, outdir=f'./out/{prompt}/')
    for it in trainer.epoch():
        ...


if __name__ == '__main__':
    generate_diffusion(sys.argv[-1])
