from base.trainer import ClassificationTrainer
from project.emotion_analysis_on_mahnob_hci.regression.trainer import MAHNOBRegressionTrainer, MAHNOBRegressionTrainerTrial
from base.output import ContinuousOutputHandlerNPY, ContinuousOutputHandlerNPYTrial
from base.metric import ContinuousMetricsCalculator, ContinuousMetricsCalculatorTrial
from base.output import PlotHandler, PlotHandlerTrial
from base.loss_function import L1, L2

import os
import time
import copy
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch import optim
from sklearn.metrics import accuracy_score


class MahnobResnetKnowledgeDistillationTrainer(ClassificationTrainer):

    def __init__(self, model, teacher, model_name='', save_path='', milestone=[0], modality=['frame'], fold=0,
                 max_epoch=2000, criterion=None, learning_rate=0.001, device='cpu', num_classes=6, patience=20,
                 early_stopping=100, factor=0.1, verbose=True, min_learning_rate=1e-5, load_best_at_each_epoch=False,
                 **kwargs):

        super().__init__(model, model_name, save_path, milestone, modality, fold, max_epoch, criterion, learning_rate,
                         device, num_classes, patience, early_stopping, factor, verbose, min_learning_rate,
                         load_best_at_each_epoch, **kwargs)

        self.teacher = teacher.to(device)

    def fit(self, dataloaders_dict, num_epochs=10, topk_accuracy=1, min_num_epochs=0,
            parameter_controller=None, checkpoint_controller=None, save_model=False):

        if self.verbose:
            print("-------")
            print("Starting training, on device:", self.device)

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'acc': 0,
            'epoch': 0,
            'kappa': -1,
        }

        start_epoch = self.start_epoch

        for epoch in np.arange(start_epoch, num_epochs):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            time_epoch_start = time.time()

            if epoch == 0 or parameter_controller.get_current_lr() < self.min_learning_rate:

                parameter_controller.release_param()
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

                if parameter_controller.early_stop:
                    break

            print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            train_loss, train_acc, train_kappa, train_confusion_matrix = self.train(
                dataloaders_dict['train'], topk_accuracy)
            validate_loss, validate_acc, validate_kappa, validate_confusion_matrix = self.validate(
                dataloaders_dict['validate'], topk_accuracy)

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)
            self.train_accuracies.append(train_acc)
            self.validate_accuracies.append(validate_acc)
            self.train_kappas.append(train_kappa)
            self.validate_kappas.append(validate_kappa)
            self.train_confusion_matrices.append(train_confusion_matrix)
            self.validate_confusion_matrices.append(validate_confusion_matrix)

            improvement = False
            if train_loss < self.best_epoch_info['loss']:
                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'acc': validate_acc,
                    'kappa': validate_kappa,
                    'epoch': epoch,
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': validate_loss,
                        'train_acc': train_acc,
                        'val_acc': validate_acc,
                    }
                }
                current_save_path = os.path.join(self.save_path,
                                                 "model_state_dict" + "_" + str(self.best_epoch_info['acc']) + ".pth")
                if save_model:
                    torch.save(self.model.state_dict(), current_save_path)

            if validate_loss < 0:
                print('\nVal loss negative!\n')
                break

            if self.verbose:
                print(
                    "Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f}, acc={:.3f}, kappa={:.3f} | Val loss={:.3f}, acc={:.3f}, kappa={:.3f} | LR={:.1e} | best={} | best_acc={:.3f} | best_kappa={:.3f} | release_count={:2} | improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        train_acc,
                        train_kappa,
                        validate_loss,
                        validate_acc,
                        validate_kappa,
                        self.optimizer.param_groups[0]['lr'],
                        int(self.best_epoch_info['epoch']) + 1,
                        self.best_epoch_info['acc'],
                        self.best_epoch_info['kappa'],
                        parameter_controller.release_count,
                        improvement,
                        self.early_stopping_counter)
                )

            checkpoint_controller.save_log_to_csv(epoch)

            if self.early_stopping and self.start_epoch > min_num_epochs:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            if isinstance(self.criterion, L2):
                self.scheduler.step(validate_loss)
            else:
                self.scheduler.step(validate_acc)

            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

    def loop(self, data_loader, train_mode=True, topk_accuracy=1):

        running_loss = 0.0
        y_true = []
        y_pred = []

        for batch_index, (X, _, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):

            inputs_frame = X['frame'].to(self.device)
            inputs_eeg = X['eeg_image'].to(self.device)
            if train_mode:
                self.optimizer.zero_grad()

            knowledge_student = self.model(inputs_eeg)
            knowledge_teacher = self.teacher(inputs_frame)

            loss = self.criterion['hint'](knowledge_student, knowledge_teacher)

            running_loss += loss.item() * self.num_classes

            if train_mode:
                loss.backward()
                self.optimizer.step()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 0
        epoch_kappa = 0
        epoch_confusion_matrix = 0

        return epoch_loss, epoch_acc, epoch_kappa, epoch_confusion_matrix


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
            # loss_soft_target = self.criterion['kd'](knowledge_student['logits'], knowledge_teacher['logits'])
            # loss_cc = self.criterion['cc'](knowledge_student['temporal'], knowledge_teacher['temporal'])
            loss_res = self.criterion['hint'](knowledge_student['res_fm'], knowledge_teacher['res_fm'])
            loss_tcn = self.criterion['hint'](knowledge_student['temporal'], knowledge_teacher['temporal'])
            loss = loss_ccc + loss_res + loss_tcn
            # loss = self.alpha * loss_ccc + (1 - self.alpha) * loss_soft_target + self.beta * loss_cc
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


class MAHNOBRegressionTrainerLoadKnowledge(MAHNOBRegressionTrainer):
    def __init__(self, model, model_name='2d1d', save_path=None, max_epoch=100, early_stopping=30, kd_weight=50, ccc_weight=1,
                 criterion=None, milestone=[0], patience=10, factor=0.1, learning_rate=0.00001, device='cpu',
                 emotional_dimension=['Valence'], metrics=None, verbose=False, print_training_metric=False,
                 load_best_at_each_epoch=False, save_plot=0, **kwargs):

        super().__init__(model, model_name, save_path, max_epoch, early_stopping, criterion, milestone, patience,
                         factor, learning_rate, device, emotional_dimension, metrics, verbose, print_training_metric,
                         load_best_at_each_epoch, save_plot, **kwargs)

        self.kd_weight = kd_weight
        self.ccc_weight = ccc_weight

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

            if 'eeg_psd' in X:
                inputs = X['eeg_psd'].to(self.device)


            knowledges = X['knowledge'].to(self.device)

            labels = torch.squeeze(Y.float().to(self.device), dim=2)

            # Determine the weight for loss function
            if train_mode:
                self.optimizer.zero_grad()

            outputs, features = self.model(inputs)

            output_handler.place_clip_output_to_subjectwise_dict(outputs.detach().cpu().numpy(), absolute_indices,
                                                                 sessions)
            continuous_label_handler.place_clip_output_to_subjectwise_dict(
                labels.detach().cpu().numpy()[:, :, np.newaxis], absolute_indices, sessions)

            loss_ccc = self.criterion['ccc'](outputs, labels.unsqueeze(2))

            if train_mode:
                loss_kd = self.criterion['kd'](knowledges, features['temporal']) * self.kd_weight
                loss = self.ccc_weight * loss_ccc + loss_kd
                # loss = loss_ccc
            else:
                loss_kd = self.criterion['kd'](knowledges, features['temporal']) * self.kd_weight
                loss = self.ccc_weight * loss_ccc + loss_kd
                # loss = loss_ccc

            running_loss += loss.mean().item()

            if train_mode:
                loss.backward()
                self.optimizer.step()

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

class MAHNOBRegressionTrainerLoadKnowledgeTrial(MAHNOBRegressionTrainerTrial):
    def __init__(self, model, model_name='2d1d', save_path=None, max_epoch=100, early_stopping=30, kd_weight=50, ccc_weight=1,
                 criterion=None, milestone=[0], patience=10, factor=0.1, learning_rate=0.00001, device='cpu',
                 emotional_dimension=['Valence'], metrics=None, verbose=False, print_training_metric=False,
                 load_best_at_each_epoch=False, save_plot=0, **kwargs):

        super().__init__(model, model_name, save_path, max_epoch, early_stopping, criterion, milestone, patience,
                         factor, learning_rate, device, emotional_dimension, metrics, verbose, print_training_metric,
                         load_best_at_each_epoch, save_plot, **kwargs)

        self.kd_weight = kd_weight
        self.ccc_weight = ccc_weight

    def loop(self, data_loader, epoch, train_mode=True):
        running_loss = 0.0

        output_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)
        continuous_label_handler = ContinuousOutputHandlerNPYTrial(self.emotional_dimension)

        # This object calculate the metrics, usually by root mean square error, pearson correlation
        # coefficient, and concordance correlation coefficient.
        metric_handler = ContinuousMetricsCalculatorTrial(self.metrics, self.emotional_dimension,
                                                     output_handler, continuous_label_handler)
        total_batch_counter = 0
        for batch_index, (X, Y, indices, trials, lengths) in tqdm(enumerate(data_loader), total=len(data_loader)):

            total_batch_counter += len(trials)

            if 'frame' in X:
                inputs = X['frame'].to(self.device)

            if 'eeg_image' in X:
                inputs = X['eeg_image'].to(self.device)

            if 'eeg_raw' in X:
                inputs = X['eeg_raw'].to(self.device)

            if 'eeg_psd' in X:
                inputs = X['eeg_psd'].to(self.device)

            #if train_mode:
            knowledges = X['knowledge'].to(self.device)

            labels = torch.squeeze(Y.float().to(self.device), dim=2)

            # Determine the weight for loss function
            if train_mode:
                loss_weights = torch.ones([labels.shape[0], labels.shape[1], 1]).to(self.device)
                self.optimizer.zero_grad()

            outputs, features = self.model(inputs)
            # outputs = self.model(knowledges)

            output_handler.update_output_for_seen_trials(outputs.detach().cpu().numpy(), trials, indices, lengths)
            continuous_label_handler.update_output_for_seen_trials(labels.detach().cpu().numpy()[:, :, np.newaxis], trials, indices, lengths)

            loss_ccc = self.criterion['ccc'](outputs, labels.unsqueeze(2))

            if train_mode:
                loss_kd = self.criterion['kd'](knowledges, features['temporal']) * self.kd_weight
                loss = self.ccc_weight * loss_ccc + loss_kd
                # loss = loss_ccc
            else:
                loss_kd = self.criterion['kd'](knowledges, features['temporal']) * self.kd_weight
                loss = self.ccc_weight * loss_ccc + loss_kd
                # loss = loss_ccc

            running_loss += loss.mean().item()

            if train_mode:
                loss.backward()
                self.optimizer.step()

            #  print_progress(batch_index, len(data_loader))

        epoch_loss = running_loss / total_batch_counter

        output_handler.average_trial_wise_records()
        continuous_label_handler.average_trial_wise_records()

        output_handler.concat_records()
        continuous_label_handler.concat_records()


        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        metric_handler.calculate_metrics()
        epoch_result_dict = metric_handler.metric_record_dict

        if self.save_plot:
            # This object plot the figures and save them.
            plot_handler = PlotHandlerTrial(self.metrics, self.emotional_dimension, epoch_result_dict,
                                       output_handler.trialwise_records, continuous_label_handler.trialwise_records,
                                       epoch=epoch, train_mode=train_mode,
                                       directory_to_save_plot=self.save_path)
            plot_handler.save_output_vs_continuous_label_plot()

        return epoch_loss, epoch_result_dict


class MAHNOBFeatureExtractorTrainer(MAHNOBRegressionTrainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def validate(self, data_loader, length_to_track, feature_save_path):
        with torch.no_grad():
            self.model.eval()
            self.loop(data_loader, length_to_track, feature_save_path, train_mode=False)

    def loop(self, data_loader, length_to_track, feature_save_path, train_mode=True):
        running_loss = 0.0

        total_batch_counter = 0

        for batch_index, (X, _, absolute_indices, sessions) in tqdm(enumerate(data_loader), total=len(data_loader)):

            total_batch_counter += len(sessions)

            if 'frame' in X:
                inputs = X['frame'].to(self.device)

            if 'eeg_image' in X:
                inputs = X['eeg_image'].to(self.device)

            os.makedirs(feature_save_path, exist_ok=True)
            save_path = os.path.join(feature_save_path, sessions[0] + ".npy")
            if not os.path.isfile(save_path):
                _, knowledge = self.model(inputs)
                temporal_knowledge = torch.squeeze(knowledge['temporal']).detach().cpu().numpy()


                np.save(save_path, temporal_knowledge)

class MAHNOBFeatureExtractorTrainerTrial(MAHNOBRegressionTrainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def validate(self, data_loader, feature_save_path):
        with torch.no_grad():
            self.model.eval()
            self.loop(data_loader, feature_save_path, train_mode=False)

    def loop(self, data_loader, feature_save_path, train_mode=True):
        running_loss = 0.0

        total_batch_counter = 0

        for batch_index, (X, _, _, sessions, _) in tqdm(enumerate(data_loader), total=len(data_loader)):

            total_batch_counter += 1

            if 'frame' in X:
                inputs = X['frame'].to(self.device)

            if 'eeg_image' in X:
                inputs = X['eeg_image'].to(self.device)

            os.makedirs(feature_save_path, exist_ok=True)
            save_path = os.path.join(feature_save_path, sessions[0] + ".npy")
            if not os.path.isfile(save_path):
                _, knowledge = self.model(inputs)
                temporal_knowledge = torch.squeeze(knowledge['temporal']).detach().cpu().numpy()


                np.save(save_path, temporal_knowledge)




