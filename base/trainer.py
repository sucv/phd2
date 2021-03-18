import torch
from torch import nn, optim


class GenericTrainer(object):

    def train(self, **kwargs):
        raise NotImplementedError

    def validate(self, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def loop(self, **kwargs):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

