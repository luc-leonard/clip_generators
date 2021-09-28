import io
from typing import Union

import requests
from omegaconf import OmegaConf
from torch import nn
import torch.nn.functional as F
import torch
import PIL.Image
from torchvision.transforms import functional as TF

from .utils import vector_quantize, clamp_with_grad
from clip_generators.models.taming_transformers.taming.models import vqgan, cond_transformer


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


class Generator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.model.eval()

    def forward(self, z):
        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

def fetch(url_or_path):
    if isinstance(url_or_path, PIL.Image.Image):
        return url_or_path
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

class ZSpace(nn.Module):
    def __init__(self, generator, image_size, device, init_image: Union[str, PIL.Image.Image], init_noise_factor):
        super().__init__()

        model = generator.model
        self.z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        self.z_min.requires_grad_(False)
        self.z_max.requires_grad_(False)

        f = 2 ** (model.decoder.num_resolutions - 1)
        n_toks = model.quantize.n_e
        toksX, toksY = image_size[0] // f, image_size[1] // f
        if init_image:
            sideX, sideY = toksX * f, toksY * f
            if isinstance(init_image, str):
                pil_image = PIL.Image.open(fetch(init_image)).convert('RGB')
            else:
                pil_image = init_image
            pil_image.thumbnail((sideX, sideY))
            self.base_image = pil_image
            base_image = TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
            self.base_image_decoded = generator(model.encode(base_image)[0])
            if init_noise_factor > 0:
                base_image = base_image * (torch.rand_like(base_image) * init_noise_factor)
            self.z, *_ = model.encode(base_image)
            self.clamp()
        else:
            self.base_image = None
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            z = one_hot @ model.quantize.embedding.weight
            self.z = z.view([-1, toksY, toksX, model.quantize.e_dim]).permute(0, 3, 1, 2)

        self.z.requires_grad_(True)

    @torch.no_grad()
    def clamp(self):
        self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

