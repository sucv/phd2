from base.experiment import GenericExperiment
from models.model import my_res50_tempool
from base.dataset import NFoldMahnobArranger, MAHNOBDataset
from base.checkpointer import ClassificationCheckpointer
from base.trainer import ClassificationTrainer
from project.emotion_classification_on_mahnob_hci.parameter_control import ParamControl
from base.utils import load_single_pkl

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

        self.stamp = args.stamp

        self.include_session_having_no_continuous_label = args.include_session_having_no_continuous_label

        self.model_name = self.experiment_name + "_" + args.model_name + "_cls_v"

        self.modality = args.modality

        self.backbone_mode = args.backbone_mode

        self.milestone = args.milestone
        self.learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        self.early_stopping = args.early_stopping
        self.patience = args.patience
        self.time_delay = args.time_delay
        self.num_epochs = args.num_epochs
        self.min_num_epochs = args.min_num_epochs
        self.factor = args.factor

        self.release_count = args.release_count
        self.device = self.init_device()

    def load_config(self):
        from project.emotion_classification_on_mahnob_hci.configs import config_mahnob as config
        return config

    def init_model(self):
        # # Here we initialize the model. It contains the spatial block and temporal block.
        # FRAME_DIM = 96
        # TIME_DEPTH = 300
        # SHARED_LINEAR_DIM1 = 1024
        # SHARED_LINEAR_DIM2 = 512
        # EMBEDDING_DIM = SHARED_LINEAR_DIM2
        # HIDDEN_DIM = 512
        # OUTPUT_DIM = 2
        # N_LAYERS = 1
        # DROPOUT_RATE_1 = 0.5
        # DROPOUT_RATE_2 = 0.5
        # model = initialize_emotion_spatial_temporal_model(
        #     self.device, frame_dim=FRAME_DIM, time_depth=TIME_DEPTH,
        #     shared_linear_dim1=SHARED_LINEAR_DIM1,
        #     shared_linear_dim2=SHARED_LINEAR_DIM2,
        #     embedding_dim=EMBEDDING_DIM,
        #     hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, n_layers=N_LAYERS,
        #     dropout_rate_1=DROPOUT_RATE_1, dropout_rate_2=DROPOUT_RATE_2
        # )

        model_eeg = my_res50_tempool(
            backbone_mode=self.backbone_mode, embedding_dim=512, output_dim=self.config['num_classes'], root_dir='')
        return model_eeg

    def init_class_label(self):

        class_labels = load_single_pkl(self.config['remote_root_directory'], "class_label")
        return class_labels

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

    def init_dataloader(self, subject_id_of_all_folds, fold_arranger, fold, class_labels):

        # Set the fold-to-partition configuration.
        # Each fold have approximately the same number of sessions.

        partition_dictionary = self.init_partition_dictionary()

        fold_index = np.roll(np.arange(self.num_folds), fold)
        subject_id_of_all_folds = list(itemgetter(*fold_index)(subject_id_of_all_folds))

        data_dict = fold_arranger.make_data_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)
        length_dict = fold_arranger.make_length_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)

        dataloaders_dict = {}
        for partition in partition_dictionary.keys():
            dataset = MAHNOBDataset(self.config, data_dict[partition], modality=self.modality,
                                    time_delay=self.time_delay, class_labels=class_labels, mode=partition)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.config['batch_size'], shuffle=True if partition == "train" else False,
                drop_last=True)

        return dataloaders_dict, length_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.model_name)

        fold_arranger = NFoldMahnobArranger(
            self.config, include_session_having_no_continuous_label=self.include_session_having_no_continuous_label,
            modality=self.modality)
        subject_id_of_all_folds, _ = fold_arranger.assign_subject_to_fold(self.num_folds)
        # subject_id_of_all_folds = [[1, 6, 3], [2, 9, 16], [4, 25, 10], [5, 7, 8], [13, 14, 17], [18, 19, 20], [21, 22],
        #                            [23, 24], [27, 28], [29, 30]]
        class_labels = self.init_class_label()
        model = self.init_model()
        criterion = torch.nn.CrossEntropyLoss()

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.num_folds))

            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            dataloaders_dict, lengths_dict = self.init_dataloader(subject_id_of_all_folds, fold_arranger, fold,
                                                                  class_labels)

            trainer = ClassificationTrainer(model, model_name=self.model_name, save_path=fold_save_path,
                                            criterion=criterion, num_classes=self.config['num_classes'],
                                            device=self.device, modality=self.modality, factor=self.factor,
                                            learning_rate=self.learning_rate, fold=fold, milestone=self.milestone,
                                            patience=self.patience, early_stopping=self.early_stopping,
                                            min_learning_rate=self.min_learning_rate, samples_weight=None)

            parameter_controller = ParamControl(trainer, release_count=self.release_count,
                                                backbone_mode=self.backbone_mode)
            checkpoint_controller = ClassificationCheckpointer(checkpoint_filename, trainer, parameter_controller,
                                                               resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            trainer.fit(dataloaders_dict, num_epochs=self.num_epochs, topk_accuracy=1,
                        min_num_epochs=self.min_num_epochs, save_model=True,
                        parameter_controller=parameter_controller, checkpoint_controller=checkpoint_controller)
            trainer.test(data_to_load=dataloaders_dict, topk_accuracy=1,
                         checkpoint_controller=checkpoint_controller)

            checkpoint_controller.save_checkpoint(trainer, parameter_controller, fold_save_path)
