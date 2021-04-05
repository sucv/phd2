import numpy as np
import torch
from torch import optim


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

    def compute_accuracy(self, outputs, targets, k=1):
        _, preds = outputs.topk(k, 1, True, True)
        preds = preds.t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))
        correct_k = correct[:k].view(-1).float()
        return correct_k

    def get_preds(self, outputs, k=1):
        _, preds = outputs.topk(k, 1, True, True)
        preds = preds.t()
        return preds[0]

    def calculate_confusion_matrix(self, preds, labels):
        confusion_matrix = np.zeros((self.num_classes, self.num_classes + 1))
        for label, pred in zip(labels, preds):
            confusion_matrix[label, pred] += 1

        confusion_matrix[:, -1] = np.sum(confusion_matrix, axis=1)
        confusion_matrix[:, :-1] /= confusion_matrix[:, -1]
        confusion_matrix[:, :-1] = np.around(confusion_matrix[:, :-1], decimals=3)
        return confusion_matrix

    def train(self, **kwargs):
        raise NotImplementedError

    def validate(self, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def loop(self, **kwargs):
        raise NotImplementedError

