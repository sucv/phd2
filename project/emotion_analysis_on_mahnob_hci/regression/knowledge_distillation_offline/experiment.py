from base.experiment import GenericExperiment
from models.model import my_2d1d, my_2dlstm
from models.knowledge_distillation_model import kd_2d1d, kd_res50
from base.dataset import NFoldMahnobArranger, MAHNOBDataset
from project.emotion_analysis_on_mahnob_hci.regression.checkpointer import Checkpointer
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.trainer import \
    MAHNOBRegressionKnowledgeDistillationTrainer
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.parameter_control import ParamControl

from base.loss_function import CCCLoss, SoftTarget, CC, Hint

import os
from operator import itemgetter

import numpy as np
import torch
import torch.nn
from torch.utils import data


class KnowledgeDistillationRegressionExperiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)

        self.num_folds = args.num_folds
        self.folds_to_run = args.folds_to_run
        self.include_session_having_no_continuous_label = 0

        self.stamp = args.stamp

        self.modality = args.modality
        self.model_name = self.experiment_name + "_" + args.student_model_name + "_" + "reg_v" + "_" + self.modality[
            0] + "_" + self.stamp
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

        self.window_length = args.window_length
        self.hop_size = args.hop_size
        self.continuous_label_frequency = args.continuous_label_frequency
        self.frame_size = args.frame_size
        self.crop_size = args.crop_size
        self.batch_size = 1

        self.num_classes = args.num_classes
        self.emotion_dimension = args.emotion_dimension

        self.device = self.init_device()


    def load_config(self):
        from project.emotion_analysis_on_mahnob_hci.configs import config_mahnob as config
        from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.configs import \
            config_knowledge_distillation as kd_config

        config = {
            'generic_config': config,
            'kd_config': kd_config
        }

        return config

    def create_model(self):

        teach_model_state_folder = self.config['kd_config']['2d1d']['teacher_frame_model_state_folder']
        student_backbone_state_folder = self.config['kd_config']['2d1d']['student_eeg_image_backbone_state_folder']

        teacher = kd_res50(backbone_state_dict=self.backbone_state_dict_frame, backbone_mode=self.backbone_mode,
                          modality=['frame'], embedding_dim=self.cnn1d_embedding_dim, channels=self.cnn1d_channels,
                          output_dim=1, kernel_size=self.cnn1d_kernel_size, dropout=self.cnn1d_dropout,
                          root_dir=self.model_load_path, folder=teach_model_state_folder, role="teacher")

        student = kd_res50(backbone_state_dict=self.backbone_state_dict_eeg, backbone_mode=self.backbone_mode,
                          modality=['eeg_image'],
                          embedding_dim=self.cnn1d_embedding_dim, channels=self.cnn1d_channels,
                          output_dim=1, kernel_size=self.cnn1d_kernel_size, dropout=self.cnn1d_dropout,
                          root_dir=self.model_load_path, folder=student_backbone_state_folder, role="student")

        return student, teacher

    def init_partition_dictionary(self):
        if self.num_folds == 3:
            partition_dictionary = {'train': 1, 'validate': 1, 'test': 1}
        elif self.num_folds == 5:
            partition_dictionary = {'train': 3, 'validate': 1, 'test': 1}
        elif self.num_folds == 9:
            partition_dictionary = {'train': 6, 'validate': 2, 'test': 1}
        elif self.num_folds == 10:
            partition_dictionary = {'train': 7, 'validate': 2, 'test': 1}
        else:
            raise ValueError("The fold number is not supported or realistic!")

        return partition_dictionary

    def init_dataloader(self, subject_id_of_all_folds, fold_arranger, fold, class_labels=None):

        # Set the fold-to-partition configuration.
        # Each fold have approximately the same number of sessions.

        partition_dictionary = self.init_partition_dictionary()

        fold_index = np.roll(np.arange(self.num_folds), fold)
        subject_id_of_all_folds = list(itemgetter(*fold_index)(subject_id_of_all_folds))

        data_dict = fold_arranger.make_data_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)
        length_dict = fold_arranger.make_length_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)

        dataloaders_dict = {}
        for partition in partition_dictionary.keys():
            dataset = MAHNOBDataset(self.config['generic_config'], data_dict[partition], modality=self.modality,
                                    emotion_dimension=self.emotion_dimension,
                                    time_delay=self.time_delay, class_labels=class_labels, mode=partition)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=True if partition == "train" else False)

        return dataloaders_dict, length_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.model_name)

        fold_arranger = NFoldMahnobArranger(dataset_load_path=self.dataset_load_path,
                                            dataset_folder=self.dataset_folder,
                                            include_session_having_no_continuous_label=self.include_session_having_no_continuous_label,
                                            modality=self.modality)
        subject_id_of_all_folds, _ = fold_arranger.assign_subject_to_fold(self.num_folds)
        print(subject_id_of_all_folds)
        model, teacher = self.create_model()

        criterion = {'ccc': CCCLoss(), 'kd': SoftTarget(T=2), 'cc': CC(gamma=0.7, P_order=5), 'hint': Hint()}

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.num_folds))

            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            model.init(fold)
            teacher.init(fold)
            dataloaders_dict, lengths_dict = self.init_dataloader(subject_id_of_all_folds, fold_arranger, fold)

            trainer = MAHNOBRegressionKnowledgeDistillationTrainer(model.to(self.device), teacher.to(self.device), stamp=self.stamp,
                                                                   model_name=self.model_name,
                                                                   learning_rate=self.learning_rate,
                                                                   metrics=self.metrics,
                                                                   save_path=fold_save_path,
                                                                   early_stopping=self.early_stopping,
                                                                   patience=self.patience, factor=self.factor,
                                                                   load_best_at_each_epoch=self.load_best_at_each_epoch,
                                                                   milestone=self.milestone, criterion=criterion,
                                                                   alpha=self.alpha, beta=self.beta,
                                                                   verbose=True,
                                                                   device=self.device)

            parameter_controller = ParamControl(trainer, gradual_release=self.gradual_release,
                                                release_count=self.release_count, backbone_mode=self.backbone_mode)
            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            if not trainer.fit_finished:
                trainer.fit(dataloaders_dict, lengths_dict, num_epochs=self.num_epochs,
                            min_num_epoch=self.min_num_epochs,
                            save_model=True, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            if not trainer.fold_finished:
                trainer.test(dataloaders_dict, lengths_dict, checkpoint_controller)
                checkpoint_controller.save_checkpoint(trainer, parameter_controller, fold_save_path)
