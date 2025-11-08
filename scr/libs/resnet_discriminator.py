import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.resnet import *
import torchvision.models as models
from libs.resnet import *
import copy
import os
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction


class ResNetImageDiscriminator(nn.Module):
    def __init__(self, in_ch=3, proj_in=384, proj_dim=128, expansion=8, ema=True, crossattn=False, config=None):
        super(ResNetImageDiscriminator, self).__init__()

        # self.net = resnet18(pretrained=True)
        self.net = resnet18()
        # self.net = resnet50()
        self.net.fc = nn.Linear(self.net.fc.in_features, proj_dim).cuda()

        self.name = 'ResNetDiscriminator'
        self.config = config

        self.proj_in = proj_in
        self.proj_dim = proj_dim
        self.context_dim = 384

        self.ema = ema
        if self.ema:
            self.prev_param = self.parameters()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout1d(0.2)
        self.bn = nn.BatchNorm1d(proj_dim)

        self.middle_block = nn.Sequential(nn.Conv2d(proj_in, proj_in, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(proj_in), nn.ReLU(inplace=True),
                                          nn.Conv2d(proj_in, proj_in, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(proj_in), nn.ReLU(inplace=True),
                                          nn.Conv2d(proj_in, proj_dim, kernel_size=1, stride=1))
        self.proj_head_latent = nn.Linear(proj_in, proj_dim)

        self.cls = nn.Linear(proj_dim, proj_dim)
        self.cls_with_latent = nn.Linear(proj_dim * 2, proj_dim)
        self.cls2 = nn.Linear(proj_dim, 1)

    def forward(self, x, latent=None):
        if self.ema: self.momentum_update()

        h = self.net(x)

        if latent is not None:
            latent = self.middle_block(latent)
            latent = torch.flatten(self.gap(latent), 1)

            h = torch.cat((h, latent), dim=1)
            h = F.relu(F.normalize(h, dim=1))

            out = F.relu(self.bn(self.cls_with_latent(h)))

        else:
            h = F.relu(F.normalize(h, dim=1))
            out = F.relu(self.bn(self.cls(h)))

        return self.cls2(out)

    @torch.no_grad()
    def momentum_update(self, m=0.99):
        """
        Momentum update of the discrminator
        """
        for param_q, param_k in zip(self.parameters(), self.prev_param):
            param_q.data = param_k.data * m + param_q.data * (1. - m)

        self.prev_param = self.parameters()

    # noinspection PyMethodMayBeStatic
    def change_classifier(self):
        self.net.classifier = nn.Linear(self.proj_dim, 1)
        return self


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor