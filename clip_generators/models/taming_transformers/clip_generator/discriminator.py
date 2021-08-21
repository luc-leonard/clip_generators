from typing import Any, List

import clip
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import transforms

from .utils import replace_grad, MakeCutouts, DifferentiableAugmentations, resample


class Prompt:
    text: str
    weight: float = 1.


class EmbeddedText(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, image_features):
        input_normed = F.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


class AveragedEmbeddedTexts(nn.Module):
    def __init__(self, embeds, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', torch.mean(embeds, 0, keepdim=True))
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
        print(self.embed.shape)

    def forward(self, image_features):
        input_normed = F.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


class ClipDiscriminator(nn.Module):
    def __init__(self, clip_model, texts, cutn, cut_pow, device, full_image_loss=True,
                 nb_augments=3):
        super().__init__()
        self.clip = clip_model
        self.embeds = []
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])
        if cutn is not None:
            self.make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn, cut_pow=cut_pow)
        else:
            self.make_cutouts = None
        self.augmentations = DifferentiableAugmentations(cutn)
        self.nb_augments = nb_augments
        self.full_image_loss = full_image_loss

        if len(texts) > 1:
            self.embeds.append(AveragedEmbeddedTexts(torch.cat(
                [self.clip.encode_text(clip.tokenize(text.strip()).to(device)).float()
                 for text in texts]
            )).cuda())
        for text in texts:
            embed = self.clip.encode_text(clip.tokenize(text.strip()).to(device)).float()
            self.embeds.append(EmbeddedText(embed).cuda())

    def forward(self, x):
        image_features: Any
        result: List[torch.Tensor] = []

        x_cutout = self.make_cutouts(x)
        if self.full_image_loss:
            x = torch.cat([
                x_cutout
                ,
                resample(x, (self.clip.visual.input_resolution, self.clip.visual.input_resolution)),
            ])
        else:
            x = x_cutout

        xs: List[torch.Tensor] = []
        for _ in range(self.nb_augments):
            xs.append(self.augmentations(x))
        xs_tensor = torch.cat(xs)
        xs_tensor = self.normalize(xs_tensor)
        image_features = self.clip.encode_image(xs_tensor).float()
        for embed in self.embeds:
            result.append(embed(image_features))
        return result
