from omegaconf import OmegaConf
from torch import nn
import torch.nn.functional as F
import torch

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
        self.model = model.eval()

    def forward(self, z):
        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)


class ZSpace(nn.Module):
    def __init__(self, model, image_size, device):
        super().__init__()
        self.z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        self.z_min.requires_grad_(False)
        self.z_max.requires_grad_(False)

        f = 2 ** (model.decoder.num_resolutions - 1)
        n_toks = model.quantize.n_e
        toksX, toksY = image_size[0] // f, image_size[1] // f

        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        self.z = z.view([-1, toksY, toksX, model.quantize.e_dim]).permute(0, 3, 1, 2)
        self.z.requires_grad_(True)

    @torch.no_grad()
    def clamp(self):
        self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

