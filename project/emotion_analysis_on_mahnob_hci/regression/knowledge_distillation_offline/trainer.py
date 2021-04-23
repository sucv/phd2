from project.emotion_analysis_on_mahnob_hci.regression.trainer import MAHNOBRegressionTrainer
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


class MAHNOBRegressionKnowledgeDistillationTrainer(MAHNOBRegressionTrainer):
    def __init__(self, model, teacher, model_name='2d1d', save_path=None, max_epoch=100, early_stopping=30,
                 criterion=None, alpha=1, beta=1,
                 milestone=[0], patience=10, factor=0.1, learning_rate=0.00001, device='cpu',
                 emotional_dimension=['Valence'], metrics=None, verbose=False, print_training_metric=False,
                 load_best_at_each_epoch=False, save_plot=0, **kwargs):

        self.teacher = teacher

        super().__init__(model, model_name, save_path, max_epoch, early_stopping, criterion, milestone, patience,
                         factor, learning_rate, device, emotional_dimension, metrics, verbose, print_training_metric,
                         load_best_at_each_epoch, save_plot, **kwargs)

        self.alpha = alpha
        self.beta = beta

    def get_parameters(self):
        params_to_update = []

        # Student
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        # Teacher
        for name, param in self.teacher.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return params_to_update


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
                inputs_frame = X['frame'].to(self.device)

            if 'eeg_image' in X:
                inputs_eeg = X['eeg_image'].to(self.device)

            labels = torch.squeeze(Y.float().to(self.device), dim=2)

            # Determine the weight for loss function
            if train_mode:
                self.optimizer.zero_grad()

            outputs, knowledge_student = self.model(inputs_eeg)
            _, knowledge_teacher = self.teacher(inputs_frame)

            output_handler.place_clip_output_to_subjectwise_dict(outputs.detach().cpu().numpy(), absolute_indices,
                                                                 sessions)
            continuous_label_handler.place_clip_output_to_subjectwise_dict(
                labels.detach().cpu().numpy()[:, :, np.newaxis], absolute_indices, sessions)
            loss_ccc = self.criterion['ccc'](outputs, labels.unsqueeze(2))
            loss_soft_target = self.criterion['kd'](knowledge_student['logits'], knowledge_teacher['logits'])
            loss_cc = self.criterion['cc'](knowledge_student['temporal'], knowledge_teacher['temporal'])
            loss = self.alpha * loss_ccc + (1 - self.alpha) * loss_soft_target + self.beta * loss_cc
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
