from base.trainer import ClassificationTrainer

from tqdm import tqdm

from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
import torch
from torch import optim
from torch.nn import MSELoss


class Trainer(ClassificationTrainer):

    def init_optimizer_and_scheduler(self):
        self.optimizer = optim.SGD(self.get_parameters(), lr=self.learning_rate, weight_decay=0.0001, momentum=0.9)

        mode = 'max'
        if isinstance(self.criterion, MSELoss):
            mode = 'min'
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, patience=self.patience,
                                                                    factor=self.factor)

    def loop(self, data_loader, train_mode=True, topk_accuracy=1):

        running_loss = 0.0
        y_true = []
        y_pred = []

        for batch_index, (X, Y) in tqdm(enumerate(data_loader), total=len(data_loader)):

            inputs = X.to(self.device)
            labels = torch.squeeze(Y.long().to(self.device))
            if len(labels.shape) > 1:
                labels = torch.squeeze(Y.float().to(self.device))

            if train_mode:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)

            if len(inputs) == 1:
                labels = labels.unsqueeze(0)
                outputs = outputs.unsqueeze(0)

            loss = self.criterion(outputs, labels)

            if len(labels.shape) > 1:
                y_true.extend(self.get_preds(labels, topk_accuracy).cpu().numpy())
            else:
                y_true.extend(labels.data.cpu().numpy())

            y_pred.extend(self.get_preds(outputs, topk_accuracy).cpu().numpy())

            running_loss += loss.item() * self.num_classes

            if train_mode:
                loss.backward()
                self.optimizer.step()

        epoch_loss = running_loss / len(y_true)
        epoch_acc = accuracy_score(y_true, y_pred)
        epoch_kappa = cohen_kappa_score(y_true, y_pred)
        epoch_confusion_matrix = self.calculate_confusion_matrix(y_pred, y_true)

        return epoch_loss, np.round(epoch_acc.item(), 3), epoch_kappa, epoch_confusion_matrix