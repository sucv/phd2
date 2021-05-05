from base.experiment import GenericExperiment
from models.model import my_2d1d, my_2dlstm, my_temporal
from models.knowledge_distillation_model import kd_2d1d, kd_res50
from base.dataset import NFoldMahnobArranger, MAHNOBDataset
from project.emotion_analysis_on_mahnob_hci.regression.checkpointer import Checkpointer
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.trainer import \
    MAHNOBRegressionTrainerLoadKnowledge
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.parameter_control import \
    ParamControl

from base.loss_function import CCCLoss, SoftTarget, CC, Hint

import os
from operator import itemgetter

import numpy as np
import torch
import torch.nn
from torch.utils import data


class TeacherEEGLSTM(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)

        self.num_folds = args.num_folds
        self.folds_to_run = args.folds_to_run
        self.include_session_having_no_continuous_label = 0

        self.stamp = args.stamp

        self.kd_weight = args.kd_weight
        self.kd_loss_function = args.kd_loss_function
        self.kl_div_T = args.kl_div_T

        self.modality = args.modality
        self.model_name = self.experiment_name + "_" + args.model_name + "_" + "reg_v" + "_" + self.modality[
            0] + "_" + self.stamp + "_kd_weight_" + str(self.kd_weight) + "_" + self.kd_loss_function + "_" + str(self.kl_div_T)
        self.backbone_state_dict_frame = args.backbone_state_dict_frame
        self.backbone_state_dict_eeg = args.backbone_state_dict_eeg
        self.backbone_mode = args.backbone_mode

        self.window_length = args.window_length
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

        student = my_temporal(model_name=self.model_name, num_inputs=self.psd_num_inputs,
                              cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                              cnn1d_dropout_rate=self.cnn1d_dropout, embedding_dim=self.lstm_embedding_dim,
                              hidden_dim=self.lstm_hidden_dim, lstm_dropout_rate=self.lstm_dropout, bidirectional=False, output_dim=1)

        return student

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

        data_dict, _ = fold_arranger.make_data_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)
        length_dict = fold_arranger.make_length_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)

        dataloaders_dict = {}
        for partition in partition_dictionary.keys():
            dataset = MAHNOBDataset(self.config['generic_config'], data_dict[partition], modality=self.modality,
                                    emotion_dimension=self.emotion_dimension,
                                    time_delay=0, class_labels=class_labels, mode=partition,
                                    load_knowledge=True, knowledge_path=self.model_load_path,
                                    knowledge_folder=self.config['kd_config']['knowledge_folder'], fold=fold)
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
        model = self.create_model()

        if self.kd_loss_function == "mse":
            criterion = {'ccc': CCCLoss(), 'kd': Hint()}
        elif self.kd_loss_function == "kl_div":
            criterion = {'ccc': CCCLoss(), 'kd': SoftTarget(self.kl_div_T)}
        else:
            raise ValueError("Unsupported loss function!")

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.num_folds))

            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            dataloaders_dict, lengths_dict = self.init_dataloader(subject_id_of_all_folds, fold_arranger, fold)

            trainer = MAHNOBRegressionTrainerLoadKnowledge(model, stamp=self.stamp, model_name=self.model_name,
                                                           learning_rate=self.learning_rate, kd_weight=self.kd_weight,
                                                           min_learning_rate=self.min_learning_rate,
                                                           metrics=self.metrics,
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
                trainer.fit(dataloaders_dict, lengths_dict, num_epochs=self.num_epochs,
                            min_num_epoch=self.min_num_epochs,
                            save_model=True, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            if not trainer.fold_finished:
                trainer.test(dataloaders_dict, lengths_dict, checkpoint_controller)
                checkpoint_controller.save_checkpoint(trainer, parameter_controller, fold_save_path)


import sys
import argparse

if __name__ == '__main__':
    frame_size = 48
    crop_size = 40

    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-experiment_name', default="emo_kd", help='The experiment name.')
    parser.add_argument('-gpu', default=1, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance server or not?')
    parser.add_argument('-stamp', default='test_kd_50', type=str, help='To indicate different experiment instances')
    parser.add_argument('-dataset', default='mahnob_hci', type=str, help='The dataset name.')
    parser.add_argument('-modality', default=['eeg_psd'], nargs="*", help='frame, eeg_image')
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?')

    parser.add_argument('-num_folds', default=10, type=int, help="How many folds to consider?")
    parser.add_argument('-folds_to_run', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], nargs="+", type=int, help='Which fold(s) to run in this session?')

    parser.add_argument('-kd_weight', default=50, type=float, help='The weight of kd loss.')
    parser.add_argument('-kd_loss_function', default='mse', type=str, help='mse, kl_div')
    parser.add_argument('-kl_div_T', default=1, type=float, help='The temperature of KL_Divergence.')

    parser.add_argument('-dataset_load_path', default='/home/zhangsu/dataset/mahnob', type=str,
                        help='The root directory of the dataset.')  # /scratch/users/ntu/su012/dataset/mahnob
    parser.add_argument('-dataset_folder', default='compacted_{:d}'.format(frame_size), type=str,
                        help='The root directory of the dataset.')  # /scratch/users/ntu/su012/dataset/mahnob
    parser.add_argument('-model_load_path', default='/home/zhangsu/phd2/load', type=str,
                        help='The path to load the trained model.')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', default='/home/zhangsu/phd2/save', type=str,
                        help='The path to save the trained model ')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', default='/home/zhangsu/phd2', type=str,
                        help='The path to the entire repository.')
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the model?')

    # Models
    parser.add_argument('-model_name', default="1d_only", help='Model: lstm_only, 1d_only')

    parser.add_argument('-backbone_state_dict_frame', default="model_state_dict_0.86272",
                        help='The filename for the backbone state dict.')
    parser.add_argument('-backbone_state_dict_eeg', default="mahnob_reg_v",
                        help='The filename for the backbone state dict.')
    parser.add_argument('-backbone_mode', default="ir", help='Mode for resnet50 backbone: ir, ir_se')

    parser.add_argument('-cnn1d_embedding_dim', default=128, type=int,
                        help='Dimensions for temporal convolutional networks feature vectors.')
    parser.add_argument('-cnn1d_channels', default=[128, 128, 128], nargs="+", type=int,
                        help='The specific epochs to do something.')
    parser.add_argument('-cnn1d_kernel_size', default=3, type=int,
                        help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-cnn1d_dropout', default=0.1, type=float, help='The dropout rate.')

    parser.add_argument('-psd_num_inputs', default=192, type=int, help='electrodes x interest bands')
    parser.add_argument('-lstm_embedding_dim', default=128, type=int, help='Dimensions for LSTM feature vectors.')
    parser.add_argument('-lstm_hidden_dim', default=128, type=int,
                        help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-lstm_dropout', default=0.4, type=float, help='The dropout rate.')

    parser.add_argument('-learning_rate', default=1e-5, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-6, type=float, help='The minimum learning rate.')
    parser.add_argument('-num_epochs', default=10, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-min_num_epochs', default=5, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-time_delay', default=0, type=float,
                        help='The time delay between input and label, in seconds.')
    parser.add_argument('-early_stopping', default=20, type=int,
                        help='If no improvement, the number of epoch to run before halting the training')


    # Groundtruth settings
    parser.add_argument('-num_classes', default=1, type=int, help='The number of classes for the dataset.')
    parser.add_argument('-emotion_dimension', default=["Valence"], nargs="*", help='The emotion dimension to analysis.')
    parser.add_argument('-metrics', default=["rmse", "pcc", "ccc"], nargs="*", help='The evaluation metrics.')
    parser.add_argument('-save_plot', default=0, type=int,
                        help='Whether to plot the session-wise output/target or not?')

    # Dataloader settings
    parser.add_argument('-window_length', default=24, type=int, help='The length in second to windowing the data.')
    parser.add_argument('-hop_size', default=8, type=int, help='The step size or stride to move the window.')
    parser.add_argument('-continuous_label_frequency', default=4, type=int,
                        help='The frequency of the continuous label.')
    parser.add_argument('-frame_size', default=frame_size, type=int, help='The size of the images.')
    parser.add_argument('-crop_size', default=crop_size, type=int, help='The size to conduct the cropping.')
    parser.add_argument('-batch_size', default=2, type=int)

    # Scheduler and Parameter Control
    parser.add_argument('-patience', default=5, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-gradual_release', default=0, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=1, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-load_best_at_each_epoch', default=1, type=int,
                        help='Whether to load the best model state at the end of each epoch?')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    experiment_handler = TeacherEEGLSTM(args)
    experiment_handler.experiment()
