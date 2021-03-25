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


class MAHNOBRegressionTrainer(GenericTrainer):
    def __init__(self, model, n_fold=10, folds_to_run=0, model_name='2d1d', save_path=None, max_epoch=100,
                 early_stopping=30, criterion=None, milestone=[0], patience=10, learning_rate=0.00001, device='cpu',
                 emotional_dimension=['Valence'], metrics=None, verbose=False, print_training_metric=False, **kwargs):

        # The device to use.
        super().__init__(model, model_name, save_path, criterion, learning_rate, early_stopping, device, max_epoch,
                         patience, verbose, **kwargs)

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
        self.combined_record_dict = {'train': {}, 'validate': {}}
        self.train_losses = []
        self.validate_losses = []
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

            validate_loss, validate_record_dict = self.validate(data_to_load['validate'], length_to_track['validate'], epoch)

            self.combined_record_dict['validate'] = self.combine_record_dict(
                self.combined_record_dict['validate'], validate_record_dict)

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            improvement = False

            validate_ccc = np.mean(validate_record_dict['overall']['Valence']['ccc'])

            # If a lower validate loss appears.
            if validate_ccc > self.best_epoch_info['ccc']:
                if save_model:
                    torch.save(self.model.state_dict(), self.save_path)

                improvement = True
                best_epoch_info = {
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
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)
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

            output_handler.place_clip_output_to_subjectwise_dict(outputs.detach().cpu().numpy(), absolute_indices, sessions)
            continuous_label_handler.place_clip_output_to_subjectwise_dict(labels.detach().cpu().numpy()[:, :, np.newaxis], absolute_indices, sessions)
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
            list in min_record_dict.
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