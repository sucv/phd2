from base.experiment import GenericExperiment
from project.emotion_regression_on_avec2019.dataset import AVEC2019Arranger, AVEC2019Dataset
from project.emotion_regression_on_avec2019.checkpointer import Checkpointer
from project.emotion_regression_on_avec2019.trainer import AVEC2019Trainer
from project.emotion_regression_on_avec2019.model import my_2d1d, my_2dlstm
from project.emotion_regression_on_avec2019.parameter_control import ParamControl
from base.loss_function import CCCLoss

import os

import torch
import torch.nn
from torch.utils import data


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.train_country = args.tc
        self.validate_country = args.vc
        self.model = args.m
        self.stamp = args.s
        self.train_emotion = self.get_train_emotion(args.e)

        self.head = "single-headed"
        self.emotion_dimension = [self.train_emotion.capitalize()]
        if args.head == "mh":
            self.head = "multi-headed"
            self.emotion_dimension = ["Arousal", "Valence"]

        self.model_name = "avec2019_" + args.m + "_" + self.train_emotion

        self.learning_rate = args.lr
        self.patience = args.p
        self.time_delay = args.d

        # if args.model_load_path == '':
        self.gpu = args.gpu
        self.cpu = args.cpu

        self.device = self.init_device()

        self.experiment()

    def load_config(self):
        from project.emotion_regression_on_avec2019.configs import avec2019_config as config
        return config

    @staticmethod
    def get_train_emotion(option):
        if option == "a":
            emotion = "arousal"
        elif option == "v":
            emotion = "valence"
        elif option == "b":
            emotion = "both"
        else:
            raise ValueError("Unknown emotion dimension to train!")

        return emotion

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

        if self.head == "multi-headed":
            output_dim = 2
        else:
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

    def init_dataloader(self):
        arranger = AVEC2019Arranger(self.config)

        data_dict = arranger.make_data_dict(train_country=self.train_country,
                                            validate_country=self.validate_country)

        length_dict = arranger.make_length_dict(train_country=self.train_country,
                                                validate_country=self.validate_country)
        train_dataset = AVEC2019Dataset(self.config, data_dict['train'], time_delay=self.time_delay,
                                        emotion=self.train_emotion,
                                        head=self.head, mode='train')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        validate_dataset = AVEC2019Dataset(self.config, data_dict['validate'], time_delay=self.time_delay,
                                           emotion=self.train_emotion, head=self.head, mode='validate')
        validate_loader = torch.utils.data.DataLoader(
            dataset=validate_dataset, batch_size=1, shuffle=False)

        dataloader_dict = {'train': train_loader, 'validate': validate_loader}
        return dataloader_dict, length_dict

    def experiment(self):

        directory_to_save_checkpoint_and_plot = os.path.join("load", self.model_name + "_" + self.stamp)
        if self.model_save_path:
            directory_to_save_checkpoint_and_plot = os.path.join(self.model_save_path,
                                                                 self.model_name + "_" + self.stamp)

        # Load the checkpoint.
        checkpoint_keys = ['time_fit_start', 'csv_filename', 'start_epoch', 'early_stopping_counter', 'best_epoch_info',
                           'combined_train_record_dict', 'combined_validate_record_dict', 'train_losses',
                           'validate_losses',
                           'current_model_weights', 'optimizer', 'scheduler', 'param_control', 'fit_finished']
        checkpoint_filename = os.path.join(directory_to_save_checkpoint_and_plot, "checkpoint.pkl")

        criterion = CCCLoss()
        model = self.init_model("my_res50_fer+_try")
        dataloader_dict, length_dict = self.init_dataloader()

        milestone = [1000]
        trainer = AVEC2019Trainer(model, stamp=self.stamp, model_name=self.model_name, learning_rate=self.learning_rate,
                                  metrics=self.config['metrics'], model_path=self.model_save_path,
                                  train_emotion=self.train_emotion, patience=self.patience,
                                  emotional_dimension=self.emotion_dimension, head=self.head,
                                  milestone=milestone, criterion=criterion, verbose=True, device=self.device)

        checkpoint_controller = Checkpointer(checkpoint_keys, checkpoint_filename, trainer)
        checkpoint_controller.load_checkpoint()

        # If the checkpoint exists, then load the parameter controller from it.
        # Otherwise, initialize a new one.
        parameter_controller = checkpoint_controller.checkpoint['param_control']
        if not checkpoint_controller.checkpoint['param_control']:
            parameter_controller = ParamControl(trainer, release_count=8)

        trainer.fit(dataloader_dict, length_dict, num_epochs=200, early_stopping=50, min_num_epoch=0,
                    directory_to_save_checkpoint_and_plot=directory_to_save_checkpoint_and_plot, save_model=True,
                    parameter_controller=parameter_controller, checkpoint_controller=checkpoint_controller)
