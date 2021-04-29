import torch
from torch import nn
import torch.nn.functional as F
import math


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


class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred, weights=None):
        gold_mean = torch.mean(gold, 1, keepdim=True, out=None)
        pred_mean = torch.mean(pred, 1, keepdim=True, out=None)
        covariance = (gold - gold_mean) * (pred - pred_mean)
        gold_var = torch.var(gold, 1, keepdim=True, unbiased=True, out=None)
        pred_var = torch.var(pred, 1, keepdim=True, unbiased=True, out=None)
        ccc = 2. * covariance / (
                (gold_var + pred_var + torch.mul(gold_mean - pred_mean, gold_mean - pred_mean)) + 1e-08)
        ccc_loss = 1. - ccc

        if weights is not None:
            ccc_loss *= weights

        return torch.mean(ccc_loss)


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        out_s = F.log_softmax(out_s / self.T, dim=1)
        out_t = F.softmax(out_t / self.T, dim=1)
        loss = F.kl_div(out_s, out_t, reduction='sum') * self.T * self.T

        return loss


'''
CC with P-order Taylor Expansion of Gaussian RBF kernel
'''


class CC(nn.Module):
    """"
    Correlation Congruence for Knowledge Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
    """

    def __init__(self, gamma, P_order):
        super(CC, self).__init__()
        self.gamma = gamma
        self.P_order = P_order

    def forward(self, feat_s, feat_t):
        corr_mat_s = self.get_correlation_matrix(feat_s)
        corr_mat_t = self.get_correlation_matrix(feat_t)

        loss = F.mse_loss(corr_mat_s, corr_mat_t)

        return loss

    def get_correlation_matrix(self, feat):
        feat = F.normalize(feat, p=2, dim=-1)
        sim_mat = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)

        for p in range(self.P_order + 1):
            corr_mat += math.exp(-2 * self.gamma) * (2 * self.gamma) ** p / \
                        math.factorial(p) * torch.pow(sim_mat, p)

        return corr_mat


class Hint(nn.Module):
    """
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	"""

    def __init__(self):
        super(Hint, self).__init__()

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(fm_s, fm_t)

        return loss
