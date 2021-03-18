from base.utils import detect_device, select_gpu, set_cpu_thread

import json
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda


class GenericExperiment(object):
    def __init__(self, args):
        self.experiment_name = args.exp
        self.config = self.load_config()
        self.init_random_seed()

        # If None, it is usually for running on High-Performance Computing Server.
        self.gpu = None
        self.cpu = None

    def load_config(self):
        # Load the config.
        with open("config_" + self.experiment_name) as config_file:
            config = json.load(config_file)
        return config

    @staticmethod
    def init_random_seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    def init_device(self):
        device = detect_device()
        if self.gpu is not None:
            select_gpu(self.gpu)

        if self.cpu is not None:
            set_cpu_thread(self.cpu)
        return device

    def init_model(self, **kwargs):
        raise NotImplementedError

    def init_dataloader(self, **kwargs):
        raise NotImplementedError

    def experiment(self):
        raise NotImplementedError
