from base.experiment import GenericExperiment
from project.emotion_regression_on_mahnob_hci.dataset import NFoldMahnobArranger, MAHNOBDataset
from project.emotion_regression_on_mahnob_hci.checkpointer import Checkpointer
from project.emotion_regression_on_mahnob_hci.trainer import MAHNOBRegressionTrainer
from project.emotion_regression_on_mahnob_hci.model import my_2d1d, my_2dlstm
from project.emotion_regression_on_mahnob_hci.parameter_control import ParamControl

from base.experiment import GenericExperiment


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

        self.n_fold = args.n_fold
        self.folds_to_run = args.folds_to_run

        self.model = args.m
        self.stamp = args.s

        self.job = args.j
        if args.j == 0:
            self.job = "reg_v"

        self.model_name = self.experiment_name + "_" + args.m + "_" + self.job

        self.modality = args.modal

        self.learning_rate = args.lr
        self.patience = args.p
        self.time_delay = args.d

        self.model_load_path = args.model_load_path
        self.model_save_path = args.model_save_path

        self.device = self.init_device()

    def load_config(self):
        from project.emotion_regression_on_mahnob_hci.configs import config_mahnob as config
        return config

    def init_model(self, backbone_model_name):
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


        output_dim = 1

        if self.model == "2d1d":
            model = my_2d1d(backbone_model_name=backbone_model_name, feature_dim=512,
                            channels_1D=[128, 128, 128, 128, 128], output_dim=output_dim, kernel_size=5, dropout=0.1,
                            root_dir=self.model_load_path)
        elif self.model == "2dlstm":
            model = my_2dlstm(backbone_model_name=backbone_model_name, feature_dim=512, hidden_dim=256,
                              output_dim=output_dim, dropout=0.4, root_dir=self.model_load_path)
        else:
            raise ValueError("Unknown base_model!")

        return model

    def init_partition_dictionary(self):
        if self.n_fold == 3:
            partition_dictionary = {'train': 1, 'validate': 1, 'test': 1}
        elif self.n_fold == 5:
            partition_dictionary = {'train': 3, 'validate': 1, 'test': 1}
        elif self.n_fold == 9:
            partition_dictionary = {'train': 6, 'validate': 2, 'test': 1}
        elif self.n_fold == 10:
            partition_dictionary = {'train': 7, 'validate': 2, 'test': 1}
        else:
            raise ValueError("The fold number is not supported or realistic!")

        return partition_dictionary

    def init_dataloader(self, subject_id_of_all_folds, fold_arranger, fold):

        # Set the fold-to-partition configuration.
        # Each fold have approximately the same number of sessions.

        partition_dictionary = self.init_partition_dictionary()

        fold_index = np.roll(np.arange(self.n_fold), fold)
        subject_id_of_all_folds = list(itemgetter(*fold_index)(subject_id_of_all_folds))

        data_dict = fold_arranger.make_data_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)
        length_dict = fold_arranger.make_length_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)

        dataloaders_dict = {}
        for partition in partition_dictionary.keys():
            dataset = MAHNOBDataset(self.config, data_dict[partition], modality=self.modality, time_delay=self.time_delay, mode=partition)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.config['batch_size'], shuffle=True if partition == "train" else False)

        return dataloaders_dict, length_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.experiment_name + "_" + self.model_name + "_" + self.stamp)

        fold_arranger = NFoldMahnobArranger(self.config, job=self.job, modality=self.modality)
        subject_id_of_all_folds, _ = fold_arranger.assign_subject_to_fold(self.n_fold)

        model = self.init_model("state_dict_0.881")
        criterion = CCCLoss()

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.n_fold))

            fold_save_path = os.path.join(save_path, str(fold))
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            # Here we generate three crucial arguments. "file_of_all_partitions" is a dictionary saving the filename to load for training,
            # validation, and test. "length_of_all_partitions" is a dictionary saving the length (frame count) for every sessions of the subjects.
            # "clip_sample_map_of_all_partitions" is a dictionary saving the session index from which a video clip comes from. The latter
            # is used to correctly place the output from clipped and shuffled video clips in the session-wise, subject-wise, and partition-wise
            # manners, which is necessary for plotting and metric calculation.
            dataloaders_dict, lengths_dict = self.init_dataloader(subject_id_of_all_folds, fold_arranger, fold)

            milestone = [1000]
            trainer = MAHNOBRegressionTrainer(model, stamp=self.stamp, model_name=self.model_name, learning_rate=self.learning_rate,
                                      metrics=self.config['metrics'], save_path=save_path, early_stopping=20, patience=self.patience,
                                      milestone=milestone, criterion=criterion, verbose=True, device=self.device)

            parameter_controller = ParamControl(trainer, release_count=8)

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger()

            trainer.fit(dataloaders_dict, lengths_dict, num_epochs=200, min_num_epoch=0, save_model=True,
                        parameter_controller=parameter_controller, checkpoint_controller=checkpoint_controller)
