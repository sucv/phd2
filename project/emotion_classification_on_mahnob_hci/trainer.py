from base.trainer import GenericTrainer
from base.output import ContinuousOutputHandlerNPY
from base.metric import ContinuousMetricsCalculator
from base.output import PlotHandler

import os
import time
import copy
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch import optim
from sklearn.metrics import accuracy_score


class MAHNOBClassificationTrainer(GenericTrainer):
    def __init__(self, model, model_name='', save_path='', milestone=[0], modality=['frame'], fold=0, max_epoch=2000,
                 criterion=None, learning_rate=0.001, device='cpu', num_classes=6, patience=20, early_stopping=100,
                 verbose=True, min_learning_rate=1e-5, **kwargs):
        super().__init__(model, model_name=model_name, save_path=save_path, criterion=criterion, min_learning_rate=min_learning_rate,
                         learning_rate=learning_rate, device=device, num_classes=num_classes, early_stopping=early_stopping,
                         max_epoch=max_epoch, patience=patience, verbose=verbose, **kwargs)

        # The networks.
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.fold = fold
        self.milestone = milestone
        self.modality = modality

        # parameter_control
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.train_losses = []
        self.validate_losses = []
        self.train_accuracies = []
        self.validate_accuracies = []
        self.train_confusion_matrices = []
        self.validate_confusion_matrices = []
        self.test_accuracy = -1
        self.test_confusion_matrix = 0
        self.csv_filename = ''
        self.best_epoch_info = {}

    def init_optimizer_and_scheduler(self):
        # self.optimizer = optim.SGD(self.get_parameters(), lr=self.learning_rate, weight_decay=0.0001, momentum=0.9)
        self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.patience, factor=0.5)

    def train(self, data_loader, topk_accuracy):
        self.model.train()
        epoch_loss, epoch_acc, epoch_confusion_matrix = self.loop(data_loader, train_mode=True, topk_accuracy=topk_accuracy)
        return epoch_loss, epoch_acc, epoch_confusion_matrix

    def validate(self, data_loader, topk_accuracy):
        with torch.no_grad():
            self.model.eval()
            epoch_loss, epoch_acc, epoch_confusion_matrix = self.loop(data_loader, train_mode=False, topk_accuracy=topk_accuracy)
        return epoch_loss, epoch_acc, epoch_confusion_matrix

    def test(
            self,
            data_to_load,
            topk_accuracy,
            checkpoint_controller=None
    ):
        if self.verbose:
            print("------")
            print("Starting testing, on device:", self.device)

        _, self.test_accuracy, self.test_confusion_matrix = self.validate(data_to_load['test'], topk_accuracy)

        if self.verbose:
            print("Test accuracy: {:.3f}".format(self.test_accuracy))
            print("------")

        checkpoint_controller.save_log_to_csv()
        self.fold_finished = True

    def fit(self, dataloaders_dict, num_epochs=10,  topk_accuracy=1, min_num_epochs=0,
            parameter_controller=None, checkpoint_controller=None, save_model=False):
        if self.verbose:
            print("-------")
            print("Starting training, on device:", self.device)

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'acc': 0,
        }

        start_epoch = self.start_epoch

        for epoch in np.arange(start_epoch, num_epochs):

            time_epoch_start = time.time()

            # if epoch == 0 or parameter_controller.get_current_lr() < self.min_learning_rate:
            #     # if epoch in [3, 6, 9, 12, 15, 18, 21, 24]:
            #     if parameter_controller.release_count == 0:
            #         print("No more layers to release, early-stop!")
            #         break
            #
            #     parameter_controller.release_param()
            #     # self.model.load_state_dict(self.best_epoch_info['model_weights'])

            print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            train_loss, train_acc, train_confusion_matrix = self.train(dataloaders_dict['train'], topk_accuracy)
            validate_loss, validate_acc, validate_confusion_matrix = self.validate(dataloaders_dict['validate'], topk_accuracy)

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)
            self.train_accuracies.append(train_acc)
            self.validate_accuracies.append(validate_acc)
            self.train_confusion_matrices.append(train_confusion_matrix)
            self.validate_confusion_matrices.append(validate_confusion_matrix)

            improvement = False
            if validate_acc > self.best_epoch_info['acc']:
                improvement = True
                self.best_epoch_info = {
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
                current_save_path = os.path.join(self.save_path, "model_state_dict" + "_" + str(self.best_epoch_info['acc']) + ".pth")
                if save_model:
                    torch.save(self.model.state_dict(), current_save_path)

            if self.early_stopping and epoch > min_num_epochs:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
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
                        int(self.best_epoch_info['epoch']) + 1,
                        self.best_epoch_info['acc'],
                        parameter_controller.release_count,
                        improvement,
                        self.early_stopping_counter)
                )

            checkpoint_controller.save_log_to_csv(epoch)

            self.scheduler.step(validate_acc)

            self.start_epoch = epoch + 1
            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

            # self.model.load_state_dict(self.best_epoch_info['model_weights'])

        self.fit_finished = True

    def loop(self, data_loader, train_mode=True, topk_accuracy=1):

        running_loss = 0.0
        y_true = []
        y_pred = []

        # self.base_model.load_state_dict(state_dict=torch.load(self.model_path))

        for batch_index, (X, Y, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):

            if 'frame' in X:
                inputs = X['frame'].to(self.device)

            if 'eeg_image' in X:
                inputs = X['eeg_image'].to(self.device)

            labels = torch.squeeze(Y.long().to(self.device))
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)

            if train_mode:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(self.get_preds(outputs, topk_accuracy).cpu().numpy())

            running_loss += loss.item() * self.num_classes

            if train_mode:
                loss.backward()
                self.optimizer.step()

        epoch_loss = running_loss / len(y_true)
        epoch_acc = accuracy_score(y_true, y_pred)
        epoch_confusion_matrix = self.calculate_confusion_matrix(y_pred, y_true)

        return epoch_loss, np.round(epoch_acc.item(), 3), epoch_confusion_matrix

