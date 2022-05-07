from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional
from kornia.augmentation import ColorJitter, RandomSharpness, RandomGaussianNoise, RandomPerspective, RandomRotation, \
    RandomAffine, RandomElasticTransform, RandomThinPlateSpline, RandomCrop, RandomErasing, RandomResizedCrop, \
    RandomHorizontalFlip
from torch import Tensor
from torch.nn import Module

from functions import ReplaceGrad, ClampWithGrad


class Prompt(Module):
    def __init__(self,
                 embed,
                 weight=1.,
                 stop=float('-inf')):
        super().__init__()
        self.replace_grad = ReplaceGrad()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, x: Tensor):
        input_normed = torch.nn.functional.normalize(x.unsqueeze(1), dim=2)
        embed_normed = torch.nn.functional.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)  # input - embed
        dists = dists * self.weight.sign()
        return self.weight.abs() * self.replace_grad.apply(dists, torch.maximum(dists, self.stop)).mean()


class MakeCutoutsBase(Module, metaclass=ABCMeta):
    def __init__(self,
                 cut_size: int,
                 num_cut: int,
                 cut_pow: float = 1.0,
                 noise: float = 0.1,
                 processes: list = None):
        super(MakeCutoutsBase, self).__init__()
        self.cut_size = cut_size
        self.num_cut = num_cut
        self.cut_pow = cut_pow  # Not used with pooling
        self.noise = noise
        if isinstance(processes, list):
            self.process = torch.nn.Sequential(*processes)
        # Pooling
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = torch.nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    @abstractmethod
    def forward(self, x: Tensor):
        pass


class MakeCutoutsLatest(MakeCutoutsBase):
    def __init__(self,
                 cut_size,
                 num_cut,
                 cut_pow=1.,
                 noise=.1,
                 augments=None):
        # Pick your own augments & their order
        processes = []
        if augments is None:
            augments = [['Af', 'Pe', 'Ji', 'Er']]
        for item in augments[0]:
            if item == 'Ji':
                processes += [ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7)]
            elif item == 'Sh':
                processes += [RandomSharpness(sharpness=0.3, p=0.5)]
            elif item == 'Gn':
                processes += [RandomGaussianNoise(mean=0.0, std=1., p=0.5)]
            elif item == 'Pe':
                processes += [RandomPerspective(distortion_scale=0.7, p=0.7)]
            elif item == 'Ro':
                processes += [RandomRotation(degrees=15, p=0.7)]
            elif item == 'Af':
                processes += [RandomAffine(degrees=15, translate=(0.1, 0.1), shear=5, p=0.7, padding_mode='zeros',
                                           keepdim=True)]
            elif item == 'Et':
                processes += [RandomElasticTransform(p=0.7)]
            elif item == 'Ts':
                processes += [RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)]
            elif item == 'Cr':
                processes += [RandomCrop(size=(self.cut_size, self.cut_size), pad_if_needed=True,
                                         padding_mode='reflect', p=0.5)]
            elif item == 'Er':
                processes += [RandomErasing(scale=(.1, .4), ratio=(.3, 1 / .3), same_on_batch=True, p=0.7)]
            elif item == 'Re':
                processes += [RandomResizedCrop(size=(self.cut_size, self.cut_size), scale=(0.1, 1),
                                                ratio=(0.75, 1.333), cropping_mode='resample', p=0.5)]
        super(MakeCutoutsLatest, self).__init__(cut_size, num_cut, cut_pow=cut_pow, noise=noise,
                                                processes=processes)

    def forward(self, x: Tensor):
        cutouts = []
        for _ in range(self.num_cut):
            cutouts += [(self.avg_pool(x) + self.max_pool(x)) / 2]
        batch = self.process(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.num_cut, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class MakeCutoutsPoolingUpdate(MakeCutoutsBase):
    def __init__(self,
                 cut_size,
                 num_cut,
                 noise=.1,
                 cut_pow=1.):
        processes = [
            RandomAffine(degrees=15, translate=(.1, .1), p=0.7, padding_mode='border'),
            RandomPerspective(distortion_scale=0.7, p=0.7),
            ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            RandomErasing(scale=(.1, .4), ratio=(.3, 1 / .3), same_on_batch=True, p=0.7)
        ]
        super(MakeCutoutsPoolingUpdate, self).__init__(cut_size, num_cut, cut_pow=cut_pow, noise=noise,
                                                       processes=processes)

    def forward(self, x: Tensor):
        cutouts = []
        for _ in range(self.num_cut):
            cutouts += [(self.avg_pool(x) + self.max_pool(x)) / 2]
        batch = self.process(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.num_cut, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class MakeCutoutsNRUpdate(MakeCutoutsLatest):
    def forward(self, x: Tensor):
        height, width = x.shape[2:4]
        min_size = min(width, height, self.cut_size)
        max_size = min(width, height)
        cutouts = []
        for _ in range(self.num_cut):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offset_x = torch.randint(0, width - size + 1, ())
            offset_y = torch.randint(0, height - size + 1, ())
            cutouts += [x[:, :, offset_y:offset_y + size, offset_x:offset_x + size]]
        batch = self.process(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.num_cut, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class MakeCutoutsUpdate(MakeCutoutsBase):
    def __init__(self,
                 cut_size: int,
                 num_cut: int,
                 cut_pow: float = 1.0,
                 noise: float = 0.1):
        processes = [
            RandomHorizontalFlip(p=0.5),
            ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            RandomSharpness(sharpness=0.3, p=0.4),
            RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.8, padding_mode='border'),
            RandomPerspective(distortion_scale=0.2, p=0.4)
        ]
        super(MakeCutoutsUpdate, self).__init__(cut_size, num_cut, cut_pow=cut_pow, noise=noise, processes=processes)

    def forward(self, x: Tensor):
        height, width = x.shape[2:4]
        min_size = min(width, height, self.cut_size)
        max_size = min(width, height)
        cutouts = []
        for _ in range(self.num_cut):
            offset_size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offset_x = torch.randint(0, width - offset_size + 1, ())
            offset_y = torch.randint(0, height - offset_size + 1, ())
            cutouts += [x[:, :, offset_y:offset_y + offset_size, offset_x:offset_x + offset_size]]
        batch = self.process(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.num_cut, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class MakeCutoutsOriginal(MakeCutoutsBase):
    def __init__(self, cut_size, num_cut, cut_pow=1.):
        super(MakeCutoutsOriginal, self).__init__(cut_size, num_cut, cut_pow=cut_pow)
        self.clamp_grad = ClampWithGrad()

    def forward(self, x: Tensor):
        height, width = x.shape[2:4]
        min_size = min(width, height, self.cut_size)
        max_size = min(width, height)
        cutouts = []
        for _ in range(self.num_cut):
            offset_size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offset_x = torch.randint(0, width - offset_size + 1, ())
            offset_y = torch.randint(0, height - offset_size + 1, ())
            cutouts += [x[:, :, offset_y:offset_y + offset_size, offset_x:offset_x + offset_size]]
        return self.clamp_grad.apply(torch.cat(cutouts, dim=0), min=0, max=1)
