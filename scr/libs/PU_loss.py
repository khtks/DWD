import numpy
import torch
from torch import nn
from torch.autograd import Function
from tools.utils import AverageMeter
import numpy as np


class nnPUSBloss(nn.Module):
    """Loss function for PUSB learning."""

    def __init__(self, gamma=1, beta=0, config=None, IS=False):
        super(nnPUSBloss, self).__init__()
        if not 0 < config.prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = config.prior
        self.gamma = gamma
        self.beta = beta
        self.positive = 1
        self.unlabeled = -1
        self.eps = 1e-7

        self.config = config
        self.IS = IS
        self.alpha = self.config.alpha if config.alpha else 3

        self.l_outs = AverageMeter('')
        self.u_outs = AverageMeter('')

        self.l_fn = (lambda x: x.detach())
        self.u_fn = (lambda x: (1 / (1 - x)).detach())
        self.IS_fn = (lambda x: (1 / (self.prior + ((1 - self.prior) * (1.0 / x - 1.0)))).detach())

    def forward(self, out, pu_target):
        # clip the predicted value to make the following optimization problem well-defined.
        out = torch.clamp(out, min=0.001, max=0.999)

        p_indices, u_indices = torch.where(pu_target == 1.)[0], torch.where(pu_target == 0.)[0]

        out_l, out_u = out[p_indices], out[u_indices]
        self.l_outs.update(torch.mean(out_l).item())
        self.u_outs.update(torch.mean(out_u).item())

        if self.IS:
            coeff = torch.tensor([(1. if i in p_indices else self.alpha * self.IS_fn(out[i])) for i in range(len(out))]).cuda()
        else:
            coeff = torch.tensor([(1. if i in p_indices else self.u_fn(out[i])) for i in range(len(out))]).cuda()

        # positive: if positive, 1 else 0
        # unlabeled: if unlabeled, 0 else 1
        positive = torch.tensor([(1. if x == 1. else 0.) for x in pu_target])[:, None].cuda()
        unlabeled = torch.tensor([(0. if x == 1. else 1.) for x in pu_target])[:, None].cuda()
        n_positive, n_unlabeled = self.config.lbs, self.config.ubs

        y_positive = -torch.log(out)
        y_unlabeled = -torch.log(1 - out)

        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        objective = positive_risk + negative_risk
        # nnPU learning
        if negative_risk.item() < -self.beta:
            objective = positive_risk - self.beta
            x_out = -self.gamma * negative_risk
        else:
            x_out = objective

        return x_out, coeff, out_l, out_u, [p_indices, u_indices]


class nnPUloss(nn.Module):
    """Loss function for PU learning."""

    def __init__(self, gamma=1, beta=0, config=None, IS=False):
        super(nnPUloss, self).__init__()
        if not 0 < config.prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = config.prior
        self.gamma = gamma
        self.beta = beta
        self.positive = 1
        self.unlabeled = -1

        self.config = config
        self.IS = IS
        self.alpha = self.config.alpha if config.alpha else 3

        self.l_outs = AverageMeter('')
        self.u_outs = AverageMeter('')

        self.loss_fn = (lambda x: torch.sigmoid(-x))
        self.clamp = (lambda x: torch.clamp(x, 0.01, 0.99))

        self.l_fn = (lambda x: self.clamp(self.loss_fn(-x)).detach())
        self.u_fn = (lambda x: (1 / (1 - self.clamp(self.loss_fn(-x)))).detach())
        self.IS_fn = (lambda x: (1 / (self.prior + ((1 - self.prior) * (1.0 / x - 1.0)))).detach())

    def forward(self, out, pu_target):
        p_indices, u_indices = torch.where(pu_target == 1.)[0], torch.where(pu_target == 0.)[0]
        out_l, out_u = out[p_indices], out[u_indices]

        self.l_outs.update(torch.mean(self.loss_fn(-out_l)).item())
        self.u_outs.update(torch.mean(self.loss_fn(-out_u)).item())

        if self.IS:
            coeff = torch.tensor([1. if i in p_indices else self.alpha * self.IS_fn(self.loss_fn(-out[i])) for i in range(len(out))]).cuda()
        else:
            coeff = torch.tensor([1. if i in p_indices else self.u_fn(self.loss_fn(-out[i])) for i in range(len(out))]).cuda()

        y_positive = self.loss_fn(out)
        y_unlabeled = self.loss_fn(-out)

        positive = torch.tensor([(1. if x == 1. else 0.) for x in pu_target])[:, None].cuda()
        unlabeled = torch.tensor([(0. if x == 1. else 1.) for x in pu_target])[:, None].cuda()
        n_positive, n_unlabeled = self.config.lbs, self.config.ubs

        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        objective = positive_risk + negative_risk
        # nnPU learning
        if negative_risk.item() < -self.beta:
            objective = positive_risk - self.beta
            x_out = -self.gamma * negative_risk
        else:
            x_out = objective

        return x_out, coeff, out_l, out_u, [p_indices, u_indices]

