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


class MAHNOBRegressionTrainer(GenericTrainer):
    def __init__(self, model, n_fold=10, folds_to_run=0, model_name='2d1d', save_path=None, max_epoch=100,
                 early_stopping=30, criterion=None, milestone=[0], patience=10, learning_rate=0.00001, device='cpu',
                 emotional_dimension=['Valence'], metrics=None, verbose=False, print_training_metric=False, **kwargs):

        # The device to use.
        super().__init__(model=model, model_name=model_name, save_path=save_path, criterion=criterion,
                         learning_rate=learning_rate, early_stopping=early_stopping, device=device, max_epoch=max_epoch,
                         patience=patience, verbose=verbose, **kwargs)

        self.device = device

        # Whether to show the information strings.
        self.verbose = verbose

        # Whether print the metrics for training.
        self.print_training_metric = print_training_metric

        # What emotional dimensions to consider.
        self.emotional_dimension = emotional_dimension

        self.metrics = metrics

        # The learning rate, and the patience of schedule.
        self.learning_rate = learning_rate
        self.patience = patience

        # The networks.
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.model = model.to(device)

        self.init_optimizer_and_scheduler()

        # parameter_control
        self.milestone = milestone

        # For checkpoint
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.time_fit_start = None
        self.combined_record_dict = {'train': {}, 'validate': {}, 'test': {}}
        self.train_losses = []
        self.validate_losses = []
        self.test_losses = []
        self.csv_filename = None
        self.best_epoch_info = None

    def init_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.patience)

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

    def train(self, data_loader, length_to_track, epoch):
        self.model.train()
        return self.loop(data_loader, length_to_track, epoch, train_mode=True)

    def validate(self, data_loader, length_to_track, epoch):
        self.model.eval()
        return self.loop(data_loader, length_to_track, epoch, train_mode=False)

    def test(
            self,
            data_to_load,
            length_to_track,
            checkpoint_controller=None
    ):
        if self.verbose:
            print("------")
            print("Starting testing, on device:", self.device)

        test_loss, test_record_dict = self.validate(data_to_load['test'], length_to_track['test'], epoch=None)

        self.combined_record_dict['test'] = self.combine_record_dict(
            self.combined_record_dict['test'], test_record_dict)

        if self.verbose:
            print(test_record_dict['overall'])
            print("------")

        checkpoint_controller.save_log_to_csv(test_record=test_record_dict)

        self.fold_finished = True

    def fit(
            self,
            data_to_load,
            length_to_track,
            num_epochs=100,
            min_num_epoch=10,
            checkpoint_controller=None,
            parameter_controller=None,
            save_model=False
    ):

        r"""
        The function to carry out training and validation.
        :param directory_to_save_checkpoint_and_plot:
        :param clip_sample_map_to_track:
        :param data_to_load: (dict), the data in training and validation partitions.
        :param length_to_track: (dict), the corresponding length of the subjects' sessions.
        :param fold: the current fold index.
        :param clipwise_frame_number: (int), how many frames contained in a mp4 file.
        :param epoch_number: (int), how many epochs to run.
        :param early_stopping: (int), how many epochs to tolerate before stopping early.
        :param min_epoch_number: the minimum epochs to run before calculating the early stopping.
        :param checkpoint: (dict), to save the information once an epoch is done
        :return: (dict), the metric dictionary recording the output and its associate continuous labels
            as a long array for each subject.
        """

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'ccc': -1e10
        }

        # Loop the epochs
        for epoch in np.arange(start_epoch, num_epochs):
            if parameter_controller.get_current_lr() < 1e-6:
                # if epoch in [3, 6, 9, 12, 15, 18, 21, 24]:
                parameter_controller.release_param()

            time_epoch_start = time.time()
            print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Get the losses and the record dictionaries for training and validation.
            train_loss, train_record_dict = self.train(data_to_load['train'], length_to_track['train'], epoch)

            # Combine the record to a long array for each subject.
            self.combined_record_dict['train'] = self.combine_record_dict(
                self.combined_record_dict['train'], train_record_dict)

            validate_loss, validate_record_dict = self.validate(data_to_load['validate'], length_to_track['validate'],
                                                                epoch)

            self.combined_record_dict['validate'] = self.combine_record_dict(
                self.combined_record_dict['validate'], validate_record_dict)

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            improvement = False

            validate_ccc = np.mean(validate_record_dict['overall']['Valence']['ccc'])

            # If a lower validate loss appears.
            if validate_ccc > self.best_epoch_info['ccc']:
                if save_model:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'ccc': validate_ccc,
                    'epoch': epoch,
                    'scalar_metrics': {
                        'train_loss': train_loss,
                        'validate_loss': validate_loss,
                    },
                    'array_metrics': {
                        'train_metric_record': train_record_dict['overall'],
                        'validate_metric_record': validate_record_dict['overall']
                    }
                }

            # Early stopping controller.
            if self.early_stopping and epoch > min_num_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True
                    if self.verbose:
                        print("\nEarly Stop!!")
                    break

            if validate_loss < 0:
                print('validate loss negative')

            if self.verbose:
                print(
                    "\n Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e}  | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

                print(train_record_dict['overall'])
                print(validate_record_dict['overall'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall'])

            self.scheduler.step(validate_ccc)
            # if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.scheduler.step(validate_loss)
            # else:
            #     self.scheduler.step()

            self.start_epoch = epoch + 1
            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        self.model.load_state_dict(self.best_epoch_info['model_weights'])

        if save_model:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))

    def loop(self, data_loader, length_to_track, epoch, train_mode=True):
        running_loss = 0.0

        output_handler = ContinuousOutputHandlerNPY(length_to_track, self.emotional_dimension)
        continuous_label_handler = ContinuousOutputHandlerNPY(length_to_track, self.emotional_dimension)

        # This object calculate the metrics, usually by root mean square error, pearson correlation
        # coefficient, and concordance correlation coefficient.
        metric_handler = ContinuousMetricsCalculator(self.metrics, self.emotional_dimension,
                                                     output_handler, continuous_label_handler)
        total_batch_counter = 0
        for batch_index, (X, Y, absolute_indices, sessions) in tqdm(enumerate(data_loader), total=len(data_loader)):

            total_batch_counter += len(sessions)

            inputs = X.to(self.device)

            labels = torch.squeeze(Y.float().to(self.device), dim=2)

            # Determine the weight for loss function
            if train_mode:
                loss_weights = torch.ones([labels.shape[0], labels.shape[1], 1]).to(self.device)
                self.optimizer.zero_grad()

            outputs = self.model(inputs)

            output_handler.place_clip_output_to_subjectwise_dict(outputs.detach().cpu().numpy(), absolute_indices,
                                                                 sessions)
            continuous_label_handler.place_clip_output_to_subjectwise_dict(
                labels.detach().cpu().numpy()[:, :, np.newaxis], absolute_indices, sessions)
            loss = self.criterion(outputs, labels.unsqueeze(2)) * outputs.size(0)

            running_loss += loss.mean().item()

            if train_mode:
                loss.backward(loss_weights, retain_graph=True)
                self.optimizer.step()

            #  print_progress(batch_index, len(data_loader))

        epoch_loss = running_loss / total_batch_counter

        # Restore the output and continuous labels to its original shape, which is session-wise.
        # By which the metrics can be calculated for each session.
        # The metrics is the average across the session and subjects of a partition.
        output_handler.get_sessionwise_dict()
        continuous_label_handler.get_sessionwise_dict()

        # Restore the output and continuous labels to partition-wise, i.e., two very long
        # arrays.  It is used for calculating the metrics.
        output_handler.get_partitionwise_dict()
        continuous_label_handler.get_partitionwise_dict()

        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        metric_handler.calculate_metrics()
        epoch_result_dict = metric_handler.metric_record_dict

        # This object plot the figures and save them.
        plot_handler = PlotHandler(self.metrics, self.emotional_dimension, epoch_result_dict,
                                   output_handler.sessionwise_dict, continuous_label_handler.sessionwise_dict,
                                   epoch=epoch, train_mode=train_mode,
                                   directory_to_save_plot=self.save_path)
        plot_handler.save_output_vs_continuous_label_plot()

        return epoch_loss, epoch_result_dict

    def combine_record_dict(self, main_record_dict, epoch_record_dict):
        r"""
        Append the metric recording dictionary of an epoch to a main record dictionary.
            Each single term from epoch_record_dict will be appended to the corresponding
            list in main_record_dict.
        Therefore, the minimum terms in main_record_dict are lists, whose element number
            are the epoch number.
        """

        # If the main record dictionary is blank, then initialize it by directly copying from epoch_record_dict.
        # Since the minimum term in epoch_record_dict is list, it is available to append further.
        if not bool(main_record_dict):
            main_record_dict = epoch_record_dict
            return main_record_dict

        # Iterate the dict and append each terms from epoch_record_dict to
        # main_record_dict.
        for (subject_id, main_subject_record), (_, epoch_subject_record) \
                in zip(main_record_dict.items(), epoch_record_dict.items()):

            # Go through emotions, e.g., Arousal and Valence.
            for emotion in self.emotional_dimension:
                # Go through metrics, e.g., rmse, pcc, and ccc.
                for metric in self.metrics:
                    # Go through the sub-dictionary belonging to each subject.
                    if subject_id != "overall":
                        session_dict = epoch_record_dict[subject_id][emotion][metric]
                        for session_id in session_dict.keys():
                            main_record_dict[subject_id][emotion][metric][session_id].append(
                                epoch_record_dict[subject_id][emotion][metric][session_id][0]
                            )

                    # In addition to subject-wise records, there are one extra sub-dictionary
                    # used to store the overall metrics, which is actually the partition-wise metrics.
                    # In this sub-dictionary, the results are obtained by first concatenating all the output
                    # and continuous labels to two long arraies, and then calculate the metrics.
                    else:
                        main_record_dict[subject_id][emotion][metric].append(
                            epoch_record_dict[subject_id][emotion][metric][0]
                        )

        return main_record_dict


class MAHNOBClassificationTrainer(GenericTrainer):
    def __init__(self, model, model_name='', save_path='', milestone=[0], fold=0, max_epoch=2000,
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

        # parameter_control
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.train_losses = []
        self.validate_losses = []
        self.train_accuracies = []
        self.validate_accuracies = []
        self.test_accuracy = -1
        self.csv_filename = ''
        self.best_epoch_info = {}

    def init_optimizer_and_scheduler(self):
        # self.optimizer = optim.SGD(self.get_parameters(), lr=self.learning_rate, weight_decay=0.0001, momentum=0.9)
        self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.patience, factor=0.5)

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
        epoch_loss, epoch_acc = self.loop(data_loader, train_mode=True, topk_accuracy=topk_accuracy)
        return epoch_loss, epoch_acc

    def validate(self, data_loader, topk_accuracy):
        with torch.no_grad():
            self.model.eval()
            epoch_loss, epoch_acc = self.loop(data_loader, train_mode=False, topk_accuracy=topk_accuracy)
        return epoch_loss, epoch_acc

    def test(
            self,
            data_to_load,
            topk_accuracy,
            checkpoint_controller=None
    ):
        if self.verbose:
            print("------")
            print("Starting testing, on device:", self.device)

        _, self.test_accuracy = self.validate(data_to_load['test'], topk_accuracy)

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

            train_loss, train_acc = self.train(dataloaders_dict['train'], topk_accuracy)
            validate_loss, validate_acc = self.validate(dataloaders_dict['validate'], topk_accuracy)

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)
            self.train_accuracies.append(train_acc)
            self.validate_accuracies.append(validate_acc)

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

        for batch_index, (_, X, Y, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = X.to(self.device)

            labels = torch.squeeze(Y.long().to(self.device))

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

        return epoch_loss, np.round(epoch_acc.item(), 3)

