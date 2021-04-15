import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, target):
        q = self.softmax(input)
        logq = torch.log(q)
        loss = - target * logq
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class CCCLoss(object):

    def __call__(self, gold, pred):
        gold_mean = torch.mean(gold, 1, keepdim=True, out=None)
        pred_mean = torch.mean(pred, 1, keepdim=True, out=None)
        covariance = (gold - gold_mean) * (pred - pred_mean)
        gold_var = torch.var(gold, 1, keepdim=True, unbiased=True, out=None)
        pred_var = torch.var(pred, 1, keepdim=True, unbiased=True, out=None)
        ccc = 2. * covariance / (
                (gold_var + pred_var + torch.mul(gold_mean - pred_mean, gold_mean - pred_mean)) + 1e-08)
        ccc_loss = 1. - ccc

        return ccc_loss
