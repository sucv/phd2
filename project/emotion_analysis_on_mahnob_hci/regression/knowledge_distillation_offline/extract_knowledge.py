from base.experiment import GenericExperiment
from models.model import my_2d1d, my_2dlstm
from models.knowledge_distillation_model import kd_2d1d, kd_res50
from base.dataset import NFoldMahnobArranger, MAHNOBDataset
from base.checkpointer import ClassificationCheckpointer as Checkpointer
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.trainer import \
    MAHNOBFeatureExtractorTrainer
from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.parameter_control import \
    ParamControl

from base.loss_function import CCCLoss, SoftTarget, CC

import os
from operator import itemgetter

import numpy as np
import torch
import torch.nn
from torch.utils import data


class KnowledgeExtractor(GenericExperiment):
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

        self.window_length = args.window_length
        self.hop_size = args.hop_size
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

        opt = self.config['kd_config']['2d1d']
        teacher = kd_2d1d(backbone_state_dict=self.backbone_state_dict_frame, backbone_mode=self.backbone_mode,
                          modality=self.modality,
                          embedding_dim=opt['cnn1d_embedding_dim'], channels=opt['cnn1d_channels'],
                          kernel_size=opt['cnn1d_kernel_size'],
                          dropout=opt['cnn1d_dropout'], root_dir=self.model_load_path,
                          folder=opt['teacher_frame_model_state_folder'], role="teacher")

        return teacher

    def init_partition_dictionary(self):
        if self.num_folds == 3:
            partition_dictionary = {'train': 1, 'validate': 1, 'test': 1}
        elif self.num_folds == 5:
            partition_dictionary = {'train': 3, 'validate': 1, 'test': 1}
        elif self.num_folds == 9:
            partition_dictionary = {'train': 6, 'validate': 2, 'test': 1}
        elif self.num_folds == 10:
            partition_dictionary = {'train': 10, 'validate': 0, 'test': 0}
        else:
            raise ValueError("The fold number is not supported or realistic!")

        return partition_dictionary

    def init_dataloader(self, subject_id_of_all_folds, fold_arranger, fold, class_labels=None, feature_extraction=True):

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
                                    time_delay=0, class_labels=class_labels, mode=partition, feature_extraction=feature_extraction)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=False)

        return dataloaders_dict, length_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.model_name)

        fold_arranger = NFoldMahnobArranger(dataset_load_path=self.dataset_load_path,
                                            dataset_folder=self.dataset_folder,
                                            include_session_having_no_continuous_label=self.include_session_having_no_continuous_label,
                                            modality=self.modality, feature_extraction=True)
        subject_id_of_all_folds, _ = fold_arranger.assign_subject_to_fold(self.num_folds)
        print(subject_id_of_all_folds)
        model = self.create_model()

        criterion = {}

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.num_folds))

            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            model.init(fold)
            dataloaders_dict, lengths_dict = self.init_dataloader(subject_id_of_all_folds, fold_arranger, fold, feature_extraction=True)

            trainer = MAHNOBFeatureExtractorTrainer(model, model_name=self.model_name,
                                                    save_path=fold_save_path, criterion=criterion, num_classes=8,
                                                    device=self.device, )

            feature_save_path = os.path.join(self.model_load_path, self.model_name,
                                             self.config['kd_config']['2d1d']['teacher_knowledge_save_folder'],
                                             str(fold))

            trainer.validate(dataloaders_dict['train'], lengths_dict['train'], feature_save_path=feature_save_path)


import sys
import argparse

if __name__ == '__main__':
    frame_size = 48
    crop_size = 40

    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-experiment_name', default="emo_kd", help='The experiment name.')
    parser.add_argument('-gpu', default=2, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance server or not?')
    parser.add_argument('-stamp', default='test', type=str, help='To indicate different experiment instances')
    parser.add_argument('-dataset', default='mahnob_hci', type=str, help='The dataset name.')
    parser.add_argument('-modality', default=['frame'], nargs="*", help='frame, eeg_image')
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?')

    parser.add_argument('-num_folds', default=10, type=int, help="How many folds to consider?")
    parser.add_argument('-folds_to_run', default=[0,1,2,3,4,5,6,7,8,9], nargs="+", type=int, help='Which fold(s) to run in this session?')

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
    parser.add_argument('-model_name', default="2d1d", help='Model: 2d1d, 2dlstm')
    parser.add_argument('-teacher_model_name', default="2d1d", help='2d1d, 2dlstm (not trained)')
    parser.add_argument('-teacher_modality', default="visual", help='visual, eeg_image')
    parser.add_argument('-student_model_name', default="2d1d", help='2d1d, 2dlstm (not trained)')
    parser.add_argument('-student_modality', default="eeg_image", help='visual, eeg_image')
    parser.add_argument('-knowledges', default=['logit', 'hint', 'nst', 'pkt', 'cc'], nargs="*",
                        help='frame, eeg_image')

    parser.add_argument('-backbone_state_dict_frame', default="2d1d_v",
                        help='The filename for the backbone state dict.')
    parser.add_argument('-backbone_state_dict_eeg', default="mahnob_reg_v",
                        help='The filename for the backbone state dict.')
    parser.add_argument('-backbone_mode', default="ir", help='Mode for resnet50 backbone: ir, ir_se')

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

    # Dataloader settings
    parser.add_argument('-window_length', default=24, type=int, help='The length in second to windowing the data.')
    parser.add_argument('-hop_size', default=8, type=int, help='The step size or stride to move the window.')
    parser.add_argument('-continuous_label_frequency', default=4, type=int,
                        help='The frequency of the continuous label.')
    parser.add_argument('-frame_size', default=frame_size, type=int, help='The size of the images.')
    parser.add_argument('-crop_size', default=crop_size, type=int, help='The size to conduct the cropping.')
    parser.add_argument('-batch_size', default=1, type=int)

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

    experiment_handler = KnowledgeExtractor(args)
    experiment_handler.experiment()
