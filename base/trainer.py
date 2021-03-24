import torch
from torch import nn, optim


class GenericTrainer(object):

    def __init__(self,
                 model,
                 model_name="my_net",
                 save_path='',
                 criterion=None,
                 learning_rate=0.0001,
                 min_learning_rate=1e-5,
                 early_stopping=100,
                 device='cpu',
                 num_classes=2,
                 max_epoch=1000,
                 patience=10,
                 verbose=True,
                 **kwargs
                 ):
        self.device = device
        self.model = model.to(device)
        self.model_name = model_name
        self.save_path = save_path

        self.num_classes = num_classes
        self.max_epoch = max_epoch
        self.start_epoch = 0
        self.early_stopping = early_stopping
        self.early_stopping_counter = self.early_stopping
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.patience = patience
        self.criterion = criterion
        self.init_optimizer_and_scheduler()

        self.verbose = verbose

    def init_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.patience)

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        return params_to_update

    def train(self, **kwargs):
        raise NotImplementedError

    def validate(self, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def loop(self, **kwargs):
        raise NotImplementedError

