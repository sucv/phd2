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
    def __init__(self, model, model_name='2d1d', save_path=None, max_epoch=100,
                 early_stopping=30, criterion=None, milestone=[0], patience=10, factor=0.1, learning_rate=0.00001, device='cpu',
                 emotional_dimension=['Valence'], metrics=None, verbose=False, print_training_metric=False,
                 load_best_at_each_epoch=False, save_plot=0, **kwargs):

        # The device to use.
        super().__init__(model=model, model_name=model_name, save_path=save_path, criterion=criterion,
                         learning_rate=learning_rate, early_stopping=early_stopping, device=device, max_epoch=max_epoch,
                         patience=patience, factor=factor, verbose=verbose, **kwargs)

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
        self.load_best_at_each_epoch = load_best_at_each_epoch

        self.save_plot = save_plot

    def train(self, data_loader, length_to_track, epoch):
        self.model.train()
        loss, result_dict = self.loop(data_loader, length_to_track, epoch,
                                      train_mode=True)
        return loss, result_dict

    def validate(self, data_loader, length_to_track, epoch):
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(data_loader, length_to_track, epoch, train_mode=False)
        return loss, result_dict

    def extract(
            self,
            data_to_load,
            length_to_track,
            checkpoint_controller=None
    ):
        if self.verbose:
            print("------")
            print("Starting knowledge extraction, on device:", self.device)



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

        checkpoint_controller.save_log_to_csv(test_record=test_record_dict['overall'])

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

        if self.fit_finished:
            print("------")
            print("Fitting already finished, proceed to test!", self.device)
            return

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

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            if epoch in self.milestone or parameter_controller.get_current_lr() < self.min_learning_rate:

                parameter_controller.release_param()
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

                if parameter_controller.early_stop:
                    break

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

            # Early stopping controller.
            if self.early_stopping and epoch > min_num_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(validate_ccc)
            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)
        self.model.load_state_dict(self.best_epoch_info['model_weights'])

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

            if 'frame' in X:
                inputs = X['frame'].to(self.device)

            if 'eeg_image' in X:
                inputs = X['eeg_image'].to(self.device)

            if 'eeg_raw' in X:
                inputs = X['eeg_raw'].to(self.device)

            if 'eeg_psd' in X:
                inputs = X['eeg_psd'].to(self.device)

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
                loss.backward()
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

        if self.save_plot:
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
