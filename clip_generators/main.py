import sys


from clip_generators.models.guided_diffusion_hd.clip_guided import Trainer as Diffusion_trainer
import clip


def generate(prompt):
    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
    trainer = Diffusion_trainer(prompt, clip_model, outdir=f'./out/{prompt}/')
    for it in trainer.epoch():
        ...


if __name__ == '__main__':
    generate(sys.argv[-1])
