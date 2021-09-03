from typing import Any, List

import clip
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import transforms

from clip_generators.models.taming_transformers.clip_generator.utils import MakeCutouts, DifferentiableAugmentations, resample


class EmbeddedText(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, image_features):
        input_normed = F.normalize(image_features.unsqueeze(1), dim=-1)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=-1)
        dists = input_normed.sub(embed_normed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * dists


class ClipDiscriminator(nn.Module):
    def __init__(self, clip_model, texts, cutn, cut_pow, device, full_image_loss=True,
                 nb_augments=3):
        super().__init__()
        self.clip = clip_model
        self.cutn = cutn
        self.embeds = []
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])
        if cutn is not None:
            self.make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn, cut_pow=cut_pow)
        else:
            self.make_cutouts = None
        self.augmentations = DifferentiableAugmentations(cutn, full_image_loss)
        self.nb_augments = nb_augments
        self.full_image_loss = full_image_loss

        for text in texts:
            embed = self.clip.encode_text(clip.tokenize(text[0].strip()).to(device)).float()
            self.embeds.append(EmbeddedText(embed, text[1]).cuda())

    def forward(self, x, n):
        image_features: Any
        result: List[torch.Tensor] = []

        x_cutout = self.make_cutouts(x)
        xs_tensor = self.normalize(x_cutout)
        image_features = self.clip.encode_image(xs_tensor).float().view([self.cutn, n, -1])
        for embed in self.embeds:
            result.append(embed(image_features))
        return result
