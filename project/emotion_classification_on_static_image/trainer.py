import time
import copy
from tqdm import tqdm

import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
from sklearn.metrics import accuracy_score


from utils.parameter_control import GenericParamControl, GenericReduceLROnPlateau


class EmotionalStaticImgClassificationTrainer(object):
    def __init__(
            self,
            model,
            milestone=[0, 10, 20, 30, 40, 50],
            fold=0,
            model_name='CFER',
            max_epoch=2000,
            optimizer=None,
            criterion=None,
            scheduler=None,
            learning_rate=0.0001,
            device='cpu',
            num_classes=6,
            patience=20,
            samples_weight=0,
            verbose=True,
            print_training_metric=False,
            warmup_learning_rate=False,
    ):
        self.fold = fold
        self.device = device
        self.verbose = verbose
        self.model_name = model_name
        self.print_training_metric = print_training_metric
        self.num_classes = num_classes
        self.max_epoch = max_epoch
        self.model_path = 'load/' + str(model_name) + '.pth'
        self.learning_rate = learning_rate
        self.warmup_learning_rate = warmup_learning_rate
        self.model = model.to(device)
        params_to_update = self.get_parameters()

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(params_to_update, learning_rate, weight_decay=0.001, betas=(0.9, 0.999))

        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience)

        # parameter_control
        self.milestone = milestone
        self.parameter_control = GenericParamControl(model)

        self.lr_control = GenericReduceLROnPlateau(patience=patience, min_epoch=0, learning_rate=learning_rate,
                                                   milestone=self.milestone, num_release=8)

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        # if self.verbose:
        #     print("Layers with params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        #         if self.verbose:
        #             print("\t", name)
        # if self.verbose:
        #     print('\t', len(params_to_update), 'layers')
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

    def train(self, data_loader, topk_accuracy):
        self.model.train()
        return self.loop(data_loader, train_mode=True, topk_accuracy=topk_accuracy)

    def validate(self, data_loader, topk_accuracy):
        self.model.eval()
        return self.loop(data_loader, train_mode=False, topk_accuracy=topk_accuracy)

    def fit(self, dataloaders_dict, num_epochs=10, early_stopping=5, topk_accuracy=1, min_num_epoch=0,
            save_model=False):
        if self.verbose:
            print("-------")
            print("Starting training, on device:", self.device)

        time_fit_start = time.time()
        train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
        early_stopping_counter = early_stopping

        best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'acc': 0,
        }

        for epoch in range(num_epochs):
            time_epoch_start = time.time()

            if epoch in self.milestone or self.lr_control.to_release:
                self.parameter_control.release_parameters_to_update()
                self.lr_control.released = True
                self.lr_control.update_lr()
                self.lr_control.to_release = False
                self.milestone = self.lr_control.update_milestone(epoch, add_milestone=50)
                params_to_update = self.get_parameters()
                self.optimizer = optim.Adam(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001, betas=(0.9, 0.999))
                # self.optimizer = optim.SGD(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001,
                #                            momentum=0.9)
            print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            train_loss, train_acc = self.train(dataloaders_dict['train'], topk_accuracy)
            val_loss, val_acc = self.validate(dataloaders_dict['val'], topk_accuracy)

            train_losses.append(train_loss)
            test_losses.append(val_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(val_acc)

            mean_loss = np.mean(train_losses)

            improvement = False
            if val_acc > best_epoch_info['acc']:
                improvement = True
                best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': val_loss,
                    'acc': val_acc,
                    'epoch': epoch,
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
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

            if val_loss < 0:
                print('\nVal loss negative!\n')
                break

            if self.verbose:
                print(
                    "Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f}, acc={:.3f}, | Val loss={:.3f}, acc={:.3f}, | LR={:.1e} | LR={:.1e} | best={} | best_acc={} | plateau_count={:2} | improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        self.optimizer.param_groups[0]['lr'],
                        self.lr_control.learning_rate,
                        int(best_epoch_info['epoch']) + 1,
                        best_epoch_info['acc'],
                        self.lr_control.plateau_count,
                        improvement,
                        early_stopping_counter)
                )

            # if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.scheduler.step(val_loss)
            #
            # else:
            #     self.scheduler.step()

            self.model.load_state_dict(best_epoch_info['model_weights'])

            if self.print_training_metric:
                print()
                time_elapsed = time.time() - time_fit_start
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

                print('Epoch with lowest val loss:', best_epoch_info['epoch'])
                for m in best_epoch_info['metrics']:
                    print('{}: {:.5f}'.format(m, best_epoch_info['metrics'][m]))
                print()

            if save_model:
                torch.save(self.model.state_dict(), self.model_path)

            self.lr_control.step(epoch, val_loss)

            if self.lr_control.updated:
                params_to_update = self.get_parameters()
                self.optimizer = optim.Adam(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001, betas=(0.9, 0.999))
                # self.optimizer = optim.SGD(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001,
                #                            momentum=0.9)
                self.lr_control.updated = False

            if self.lr_control.halt:
                break

    def loop(self, data_loader, train_mode=True, topk_accuracy=1):

        running_loss = 0.0
        running_corrects = 0
        total_data_count = 0
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

            # print_progress(batch_index, len(data_loader))

        epoch_loss = running_loss / len(y_true)
        epoch_acc = accuracy_score(y_true, y_pred)

        return epoch_loss, np.round(epoch_acc.item(), 3)
