from base.trainer import GenericTrainer
from project.emotion_classification_on_static_image.parameter_control import ParamControl

import time
import copy
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
import torch.utils.data
from sklearn.metrics import accuracy_score


class ImageClassificationTrainer(GenericTrainer):
    def __init__(self, model, milestone=[0], fold=0, model_name='', max_epoch=2000, optimizer=None,
                 criterion=None, learning_rate=0.0001, device='cpu', num_classes=6, patience=20,
                 verbose=True, **kwargs):
        super().__init__(model, model_name, optimizer, criterion, learning_rate, device, num_classes,
                         max_epoch, patience, verbose, **kwargs)
        self.fold = fold

        # parameter_control
        self.milestone = milestone

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

    def train(self, data_loader, topk_accuracy):
        self.model.train()
        return self.loop(data_loader, train_mode=True, topk_accuracy=topk_accuracy)

    def validate(self, data_loader, topk_accuracy):
        self.model.eval()
        return self.loop(data_loader, train_mode=False, topk_accuracy=topk_accuracy)

    def fit(self, dataloaders_dict, num_epochs=10, early_stopping=5, topk_accuracy=1, min_num_epoch=0,
            parameter_controller=None, save_model=False):
        if self.verbose:
            print("-------")
            print("Starting training, on device:", self.device)

        time_fit_start = time.time()
        train_losses, validate_losses, train_accuracies, validate_accuracies = [], [], [], []
        early_stopping_counter = early_stopping

        best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'acc': 0,
        }

        for epoch in range(num_epochs):
            time_epoch_start = time.time()

            if parameter_controller.get_current_lr() < 1e-7:
                # if epoch in [3, 6, 9, 12, 15, 18, 21, 24]:
                parameter_controller.release_param()

            print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            train_loss, train_acc = self.train(dataloaders_dict['train'], topk_accuracy)
            validate_loss, validate_acc = self.validate(dataloaders_dict['validate'], topk_accuracy)

            train_losses.append(train_loss)
            validate_losses.append(validate_loss)
            train_accuracies.append(train_acc)
            validate_accuracies.append(validate_acc)

            improvement = False
            if validate_acc > best_epoch_info['acc']:
                improvement = True
                best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'acc': validate_acc,
                    'epoch': epoch,
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': validate_loss,
                        'train_acc': train_acc,
                        'val_acc': validate_acc,
                    }
                }

            if early_stopping and epoch > min_num_epoch:
                if improvement:
                    early_stopping_counter = early_stopping
                else:
                    early_stopping_counter -= 1

                if early_stopping_counter <= 0:
                    if self.verbose:
                        print("\nEarly Stop!\n")
                    break

            if validate_loss < 0:
                print('\nVal loss negative!\n')
                break

            if self.verbose:
                print(
                    "Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f}, acc={:.3f}, | Val loss={:.3f}, acc={:.3f}, | LR={:.1e} | best={} | best_acc={} | release_count={:2} | improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        train_acc,
                        validate_loss,
                        validate_acc,
                        self.optimizer.param_groups[0]['lr'],
                        int(best_epoch_info['epoch']) + 1,
                        best_epoch_info['acc'],
                        parameter_controller.release_count,
                        improvement,
                        early_stopping_counter)
                )

            self.scheduler.step(validate_acc)

            self.model.load_state_dict(best_epoch_info['model_weights'])

            if save_model:
                torch.save(self.model.state_dict(), self.model_path)

    def loop(self, data_loader, train_mode=True, topk_accuracy=1):

        running_loss = 0.0
        y_true = []
        y_pred = []

        # self.base_model.load_state_dict(state_dict=torch.load(self.model_path))

        for batch_index, (X, Y) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = X.to(self.device)

            labels = torch.squeeze(Y.long().to(self.device))

            if train_mode:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels) * outputs.size(0)

            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(self.get_preds(outputs, topk_accuracy).cpu().numpy())

            running_loss += loss.item() * self.num_classes

            if train_mode:
                loss.backward()
                self.optimizer.step()

        epoch_loss = running_loss / len(y_true)
        epoch_acc = accuracy_score(y_true, y_pred)

        return epoch_loss, np.round(epoch_acc.item(), 3)
