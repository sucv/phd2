from base.experiment import GenericExperiment
from models.model import my_2d1d, my_2dlstm, my_eeg1d, my_temporal, my_eeglstm
from base.dataset import NFoldMahnobArrangerTrial, MAHNOBDatasetTrial
from project.emotion_analysis_on_mahnob_hci.regression.checkpointer import Checkpointer
from project.emotion_analysis_on_mahnob_hci.regression.trainer import MAHNOBRegressionTrainerTrial
from project.emotion_analysis_on_mahnob_hci.regression.parameter_control import ParamControl
from base.utils import load_single_pkl
from base.loss_function import CCCLoss

import os
from operator import itemgetter

import numpy as np
import torch
import torch.nn
from torch.utils import data


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)

        self.num_folds = args.num_folds
        self.folds_to_run = args.folds_to_run
        self.include_session_having_no_continuous_label = 0
        self.normalize_eeg_raw = args.normalize_eeg_raw
        self.stamp = args.stamp

        self.case = args.case

        self.modality = args.modality
        self.model_name = self.experiment_name + "_" + args.model_name + "_" + "reg_v" + "_" + self.modality[
            0] + "_" + self.case + "_" + self.stamp

        self.backbone_state_dict_frame = args.backbone_state_dict_frame
        self.backbone_state_dict_eeg = args.backbone_state_dict_eeg
        self.backbone_mode = args.backbone_mode

        self.cnn1d_embedding_dim = args.cnn1d_embedding_dim
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout

        self.eegnet_num_channels = args.eegnet_num_channels
        self.eegnet_num_samples = args.eegnet_num_samples
        self.eegnet_dropout_rate = args.eegnet_dropout_rate
        self.eegnet_kernel_length = args.eegnet_kernel_length
        self.eegnet_kernel_length2 = args.eegnet_kernel_length2
        self.eegnet_F1 = args.eegnet_F1
        self.eegnet_F2 = args.eegnet_F2
        self.eegnet_D = args.eegnet_D
        self.eegnet_window_sec = args.eegnet_window_sec
        self.eegnet_stride_sec = args.eegnet_stride_sec

        self.psd_num_inputs = args.psd_num_inputs

        self.window_sec = args.window_sec
        self.hop_size_sec = args.hop_size_sec
        self.continuous_label_frequency = args.continuous_label_frequency
        self.frame_size = args.frame_size
        self.crop_size = args.crop_size
        self.batch_size = args.batch_size

        self.milestone = args.milestone
        self.learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        self.early_stopping = args.early_stopping
        self.patience = args.patience
        self.time_delay = args.time_delay
        self.num_epochs = args.num_epochs
        self.min_num_epochs = args.min_num_epochs
        self.factor = args.factor
        self.gradual_release = args.gradual_release
        self.release_count = args.release_count
        self.load_best_at_each_epoch = args.load_best_at_each_epoch

        self.num_classes = args.num_classes
        self.emotion_dimension = args.emotion_dimension
        self.metrics = args.metrics

        self.save_plot = args.save_plot

        self.device = self.init_device()

    def load_config(self):
        from project.emotion_analysis_on_mahnob_hci.configs import config_mahnob as config
        return config

    def create_model(self):

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        output_dim = 1
        if "eeg_image" in self.modality:
            backbone_state_dict = self.backbone_state_dict_eeg

        if "frame" in self.modality:
            backbone_state_dict = self.backbone_state_dict_frame

        if "2d1d" in self.model_name:
            model = my_2d1d(backbone_state_dict=backbone_state_dict, backbone_mode=self.backbone_mode,
                            embedding_dim=self.cnn1d_embedding_dim, channels=self.cnn1d_channels,
                            modality=self.modality, output_dim=output_dim, kernel_size=self.cnn1d_kernel_size,
                            dropout=self.cnn1d_dropout, root_dir=self.model_load_path)
        elif "2dlstm" in self.model_name:
            model = my_2dlstm(backbone_state_dict=backbone_state_dict, backbone_mode=self.backbone_mode,
                              embedding_dim=self.lstm_embedding_dim, hidden_dim=self.lstm_hidden_dim,
                              modality=self.modality,
                              output_dim=output_dim, dropout=self.lstm_dropout,
                              root_dir=self.model_load_path)
        elif "eegnet1d" in self.model_name:
            model = my_eeg1d(num_channels=self.eegnet_num_channels, num_samples=self.eegnet_num_samples,
                             dropout_rate=self.eegnet_dropout_rate,
                             kernel_length=self.eegnet_kernel_length, kernel_length2=self.eegnet_kernel_length2,
                             F1=self.eegnet_F1, F2=self.eegnet_F2,
                             D=self.eegnet_D, cnn1d_channels=self.cnn1d_channels,
                             cnn1d_kernel_size=self.cnn1d_kernel_size, cnn1d_dropout_rate=self.cnn1d_dropout,
                             output_dim=output_dim)
        elif "eegnetlstm" in self.model_name:
            model = my_eeglstm(num_channels=self.eegnet_num_channels, num_samples=self.eegnet_num_samples,
                               dropout_rate=self.eegnet_dropout_rate,
                               kernel_length=self.eegnet_kernel_length, kernel_length2=self.eegnet_kernel_length2,
                               F1=self.eegnet_F1, F2=self.eegnet_F2,
                               D=self.eegnet_D, embedding_dim=self.lstm_embedding_dim, hidden_dim=self.lstm_hidden_dim,
                               lstm_dropout_rate=self.lstm_dropout)
        elif "1d_only" in self.model_name or "lstm_only" in self.model_name:
            model = my_temporal(model_name=self.model_name, num_inputs=self.psd_num_inputs,
                                cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                                cnn1d_dropout_rate=self.cnn1d_dropout, embedding_dim=self.lstm_embedding_dim,
                                hidden_dim=self.lstm_hidden_dim, lstm_dropout_rate=self.lstm_dropout,
                                output_dim=output_dim)
        else:
            raise ValueError("Unsupported model!")
        return model

    def init_partition_setting(self):
        # partition_setting = {'train': 168, 'validate': 47, 'test': 24}
        partition_setting = {'train': 129, 'validate': 86, 'test': 24}
        return partition_setting

    def init_dataloader(self, partition_setting, trial_id_of_all_folds, fold_arranger, fold, class_labels=None):

        # Set the fold-to-partition configuration.
        # Each fold have approximately the same number of sessions.
        self.init_random_seed()
        trial_index = np.roll(trial_id_of_all_folds, 24 * fold)
        trial_id_of_all_partitions = fold_arranger.assign_trial_to_partition(trial_index)
        data_dict, normalize_dict = fold_arranger.make_data_dict(trial_id_of_all_partitions)

        dataloaders_dict = {}
        for partition in partition_setting.keys():
            dataset = MAHNOBDatasetTrial(self.config, data_dict[partition], normalize_dict=normalize_dict,
                                         modality=self.modality,
                                         continuous_label_frequency=self.config['frequency_dict']['continuous_label'],
                                         normalize_eeg_raw=self.normalize_eeg_raw,
                                         emotion_dimension=self.emotion_dimension,
                                         eegnet_window_sec=self.eegnet_window_sec,
                                         eegnet_stride_sec=self.eegnet_stride_sec,
                                         time_delay=self.time_delay, class_labels=class_labels, mode=partition)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=True if partition == "train" else False)

        return dataloaders_dict, normalize_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.model_name)

        partition_setting = self.init_partition_setting()

        fold_arranger = NFoldMahnobArrangerTrial(dataset_load_path=self.dataset_load_path,
                                                 normalize_eeg_raw=self.normalize_eeg_raw,
                                                 dataset_folder=self.dataset_folder, window_sec=self.window_sec,
                                                 hop_size_sec=self.hop_size_sec, partition_setting=partition_setting,
                                                 include_session_having_no_continuous_label=self.include_session_having_no_continuous_label,
                                                 modality=self.modality)

        trial_id_of_all_partitions = fold_arranger.get_trial_indices_having_continuous_label()

        criterion = CCCLoss()

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.num_folds))

            model = self.create_model()

            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            if "2d1d" in self.model_name or "2dlstm" in self.model_name:
                model.init(fold)

            dataloaders_dict, normalize_dict = self.init_dataloader(partition_setting, trial_id_of_all_partitions,
                                                                    fold_arranger, fold)

            # path = "/home/zhangsu/phd2/load/trained_2d1d_frame/2d1d_v_1.pth"
            # state_dict = torch.load(path, map_location='cpu')
            # model.load_state_dict(state_dict)

            trainer = MAHNOBRegressionTrainerTrial(model, stamp=self.stamp, model_name=self.model_name,
                                                   learning_rate=self.learning_rate,
                                                   min_learning_rate=self.min_learning_rate, metrics=self.metrics,
                                                   save_path=fold_save_path, early_stopping=self.early_stopping,
                                                   patience=self.patience, factor=self.factor,
                                                   load_best_at_each_epoch=self.load_best_at_each_epoch,
                                                   milestone=self.milestone, criterion=criterion, verbose=True,
                                                   save_plot=self.save_plot,
                                                   device=self.device)

            parameter_controller = ParamControl(trainer, gradual_release=self.gradual_release,
                                                release_count=self.release_count, backbone_mode=self.backbone_mode)
            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            if not trainer.fit_finished:
                trainer.fit(dataloaders_dict, num_epochs=self.num_epochs,
                            min_num_epoch=self.min_num_epochs,
                            save_model=True, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            if not trainer.fold_finished:
                trainer.test(dataloaders_dict, checkpoint_controller)
                checkpoint_controller.save_checkpoint(trainer, parameter_controller, fold_save_path)
