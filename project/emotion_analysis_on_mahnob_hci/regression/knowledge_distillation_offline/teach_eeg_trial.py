from base.experiment import GenericExperiment
from models.model import my_2d1d, my_2dlstm, my_temporal, my_test
from models.knowledge_distillation_model import kd_2d1d, kd_res50
from base.dataset import NFoldMahnobArrangerTrial, MAHNOBDatasetTrial
from project.emotion_analysis_on_mahnob_hci.regression.checkpointer import Checkpointer
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.trainer import \
    MAHNOBRegressionTrainerLoadKnowledgeTrial
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.parameter_control import \
    ParamControl

from base.loss_function import CCCLoss, SoftTarget, CC, L1, L2, L1S

import os
from operator import itemgetter

import numpy as np
import torch
import torch.nn
from torch.utils import data


class TeacherEEG1D(GenericExperiment):
    def __init__(self, args, ccc_weight=None, kd_weight=None):
        super().__init__(args)

        self.num_folds = args.num_folds
        self.folds_to_run = args.folds_to_run
        self.include_session_having_no_continuous_label = 0
        self.case = args.case
        self.stamp = args.stamp

        self.ccc_weight = ccc_weight
        if ccc_weight is None:
            self.ccc_weight = args.ccc_weight

        self.kd_weight = kd_weight
        if kd_weight is None:
            self.kd_weight = args.kd_weight

        self.kd_loss_function = args.kd_loss_function
        self.kl_div_T = args.kl_div_T

        self.modality = args.modality
        self.model_name = self.experiment_name + "_" + args.model_name + "_" + "reg_v" + "_" + self.modality[
            0] + "_" + self.case + "_" + self.stamp + "_ccc_weight_" + str(self.ccc_weight) + "_kd_weight_" + str(
            self.kd_weight) + "_" + self.kd_loss_function + "_" + str(
            self.kl_div_T)
        self.backbone_state_dict_frame = args.backbone_state_dict_frame
        self.backbone_state_dict_eeg = args.backbone_state_dict_eeg
        self.backbone_mode = args.backbone_mode

        self.window_sec = args.window_sec
        self.hop_size = args.hop_size
        self.continuous_label_frequency = args.continuous_label_frequency
        self.frame_size = args.frame_size
        self.crop_size = args.crop_size
        self.batch_size = args.batch_size

        self.cnn1d_embedding_dim = args.cnn1d_embedding_dim
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout
        self.psd_num_inputs = args.psd_num_inputs

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

        self.knowledge_load_path = args.knowledge_load_path + "_trial"
        self.device = self.init_device()

        if self.debug:
            self.folds_to_run = [0, 1, 2]
            self.num_folds = 3
            self.num_epochs = 1

    def load_config(self):
        from project.emotion_analysis_on_mahnob_hci.configs import config_mahnob as config
        from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.configs import \
            config_knowledge_distillation as kd_config

        if self.debug:
            from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.configs_debug import \
                config_knowledge_distillation as kd_config


        config = {
            'generic_config': config,
            'kd_config': kd_config
        }

        return config

    def create_model(self):

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        student = my_temporal(model_name=self.model_name, num_inputs=self.psd_num_inputs,
                              cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                              cnn1d_dropout_rate=self.cnn1d_dropout, embedding_dim=self.lstm_embedding_dim,
                              hidden_dim=self.lstm_hidden_dim, lstm_dropout_rate=self.lstm_dropout, bidirectional=False,
                              output_dim=1)

        # student = my_test()
        return student

    def init_partition_setting(self):
        partition_setting = {'train': 168, 'validate': 47, 'test': 24}
        if self.debug:
            partition_setting = {'train': 3, 'validate': 2, 'test': 1}
        return partition_setting

    def init_dataloader(self, partition_setting, trial_id_of_all_folds, fold_arranger, fold, class_labels=None):

        # Set the fold-to-partition configuration.
        # Each fold have approximately the same number of sessions.

        trial_index = np.roll(trial_id_of_all_folds, 24 * fold)
        trial_id_of_all_partitions = fold_arranger.assign_trial_to_partition(trial_index)
        data_dict, normalize_dict = fold_arranger.make_data_dict(trial_id_of_all_partitions)

        dataloaders_dict = {}
        self.init_random_seed()
        for partition in partition_setting.keys():
            dataset = MAHNOBDatasetTrial(self.config['generic_config'], data_dict[partition], modality=self.modality,
                                         emotion_dimension=self.emotion_dimension,
                                         time_delay=0, class_labels=class_labels, mode=partition,
                                         load_knowledge=True, knowledge_path=self.knowledge_load_path, fold=fold)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=True if partition == "train" else False)

        return dataloaders_dict, normalize_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.model_name)

        partition_setting = self.init_partition_setting()

        fold_arranger = NFoldMahnobArrangerTrial(dataset_load_path=self.dataset_load_path, window_sec=self.window_sec,
                                                 hop_size_sec=self.hop_size,
                                                 dataset_folder=self.dataset_folder,
                                                 partition_setting=partition_setting,
                                                 include_session_having_no_continuous_label=self.include_session_having_no_continuous_label,
                                                 modality=self.modality)
        trial_id_of_all_partitions = fold_arranger.get_trial_indices_having_continuous_label()

        if self.kd_loss_function == "l1":
            criterion = {'ccc': CCCLoss(), 'kd': L1()}
        elif self.kd_loss_function == "l2":
            criterion = {'ccc': CCCLoss(), 'kd': L2()}
        elif self.kd_loss_function == "l1s":
            criterion = {'ccc': CCCLoss(), 'kd': L1S()}
        else:
            raise ValueError("Unsupported loss function!")

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.num_folds))

            model = self.create_model()

            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            dataloaders_dict, normalize_dict = self.init_dataloader(partition_setting, trial_id_of_all_partitions,
                                                                    fold_arranger, fold)

            trainer = MAHNOBRegressionTrainerLoadKnowledgeTrial(model, stamp=self.stamp, model_name=self.model_name,
                                                                learning_rate=self.learning_rate,
                                                                kd_weight=self.kd_weight, ccc_weight=self.ccc_weight,
                                                                min_learning_rate=self.min_learning_rate,
                                                                metrics=self.metrics,
                                                                save_path=fold_save_path,
                                                                early_stopping=self.early_stopping,
                                                                patience=self.patience, factor=self.factor,
                                                                load_best_at_each_epoch=self.load_best_at_each_epoch,
                                                                milestone=self.milestone, criterion=criterion,
                                                                verbose=True,
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
