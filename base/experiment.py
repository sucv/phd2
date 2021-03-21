from base.utils import detect_device, select_gpu, set_cpu_thread

import json
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda


class GenericExperiment(object):
    def __init__(self, args):
        self.args = args
        self.experiment_name = args.exp
        self.model_load_path = args.model_load_path
        self.model_save_path = args.model_save_path
        self.config = self.load_config()
        self.init_random_seed()

        # If None, it is usually for running on High-Performance Computing Server.
        self.gpu = args.gpu
        self.cpu = args.cpu
        # If the code is to run on high-performance computer, which is usually not
        # available to specify gpu index and cpu threads, then set them to none.
        if self.args.hpc:
            self.gpu = None
            self.cpu = None

    def load_config(self):
        raise NotImplementedError

    @staticmethod
    def init_random_seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    def init_device(self):
        device = detect_device()

        if not self.args.hpc:
            select_gpu(self.gpu)
            set_cpu_thread(self.cpu)

        return device

    def init_model(self, **kwargs):
        raise NotImplementedError

    def init_dataloader(self, **kwargs):
        raise NotImplementedError

    def experiment(self):
        raise NotImplementedError
