from base.experiment import GenericExperiment
from models.model import my_2d1d, my_2dlstm
from project.emotion_regression_on_avec2019.dataset import AVEC2019Arranger, AVEC2019Dataset
from project.emotion_regression_on_avec2019.checkpointer import Checkpointer
from project.emotion_regression_on_avec2019.trainer import AVEC2019Trainer
from project.emotion_regression_on_avec2019.parameter_control import ParamControl
from base.loss_function import CCCLoss

import os

import torch
import torch.nn
from torch.utils import data


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.train_country = args.train_country
        self.validate_country = args.validate_country

        self.stamp = args.stamp
        self.train_emotion = self.get_train_emotion(args.train_emotion)

        self.head = "single-headed"
        self.emotion_dimension = [self.train_emotion.capitalize()]
        if args.head == "mh":
            self.head = "multi-headed"
            self.emotion_dimension = ["Arousal", "Valence"]

        self.model_name = self.experiment_name + "_" + args.model_name + "_" + self.train_emotion + "_" + self.stamp
        self.backbone_state_dict = args.backbone_state_dict
        self.backbone_mode = args.backbone_mode

        self.cnn1d_embedding_dim = args.cnn1d_embedding_dim
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout

        self.milestone = args.milestone
        self.learning_rate = args.learning_rate
        self.early_stopping = args.early_stopping
        self.patience = args.patience
        self.time_delay = args.time_delay
        self.num_epochs = args.num_epochs
        self.min_num_epochs = args.min_num_epochs
        self.factor = args.factor

        self.window_length = args.window_length
        self.hop_size = args.hop_size
        self.continuous_label_frequency = args.continuous_label_frequency
        self.frame_size = args.frame_size
        self.crop_size = args.crop_size
        self.batch_size = args.batch_size

        self.num_classes = args.num_classes
        self.emotion_dimension = args.emotion_dimension
        self.metrics = args.metrics
        self.release_count = args.release_count

        self.device = self.init_device()

    def load_config(self):
        from project.emotion_regression_on_avec2019.configs import config_avec2019 as config
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

    def init_model(self):

        if self.head == "multi-headed":
            output_dim = 2
        else:
            output_dim = 1

        if "2d1d" in self.model_name:
            model = my_2d1d(backbone_state_dict=self.backbone_state_dict, backbone_mode=self.backbone_mode,
                            embedding_dim=self.cnn1d_embedding_dim, channels=self.cnn1d_channels,
                            output_dim=output_dim, kernel_size=self.cnn1d_kernel_size,
                            dropout=self.cnn1d_dropout, root_dir=self.model_load_path)

        elif "2dlstm" in self.model_name:
            model = my_2dlstm(backbone_state_dict=self.backbone_state_dict, backbone_mode=self.backbone_mode,
                              embedding_dim=self.lstm_embedding_dim, hidden_dim=self.lstm_hidden_dim,
                              output_dim=output_dim, dropout=self.lstm_dropout,
                              root_dir=self.model_load_path)
        else:
            raise ValueError("Unknown base_model!")

        return model

    def init_dataloader(self):
        arranger = AVEC2019Arranger(self.dataset_load_path, self.dataset_folder, window_length=self.window_length,
                                    hop_size=self.hop_size, continuous_label_frequency=self.continuous_label_frequency)

        data_dict = arranger.make_data_dict(train_country=self.train_country,
                                            validate_country=self.validate_country)

        length_dict = arranger.make_length_dict(train_country=self.train_country,
                                                validate_country=self.validate_country)
        train_dataset = AVEC2019Dataset(data_dict['train'], crop_size=self.crop_size,
                                        frame_to_label_ratio=self.config['downsampling_interval_dict']['frame'],
                                        time_delay=self.time_delay,
                                        emotion=self.train_emotion, head=self.head, mode='train')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        validate_dataset = AVEC2019Dataset(data_dict['train'], crop_size=self.crop_size,
                                           frame_to_label_ratio=self.config['downsampling_interval_dict']['frame'],
                                           time_delay=self.time_delay,
                                           emotion=self.train_emotion, head=self.head, mode='validate')

        validate_loader = torch.utils.data.DataLoader(
            dataset=validate_dataset, batch_size=self.batch_size, shuffle=False)

        dataloader_dict = {'train': train_loader, 'validate': validate_loader}
        return dataloader_dict, length_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.model_name)
        os.makedirs(save_path, exist_ok=True)

        checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

        model = self.init_model()
        model.init()
        dataloader_dict, length_dict = self.init_dataloader()
        criterion = CCCLoss()

        trainer = AVEC2019Trainer(model, model_name=self.model_name, learning_rate=self.learning_rate,
                                  metrics=self.metrics, save_path=save_path, early_stopping=self.early_stopping,
                                  train_emotion=self.train_emotion, patience=self.patience, factor=self.factor,
                                  emotional_dimension=self.emotion_dimension, head=self.head,
                                  milestone=self.milestone, criterion=criterion, verbose=True, device=self.device)

        parameter_controller = ParamControl(trainer, release_count=self.release_count, backbone_mode=self.backbone_mode)

        checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

        if self.resume:
            trainer, parameter_controller = checkpoint_controller.load_checkpoint()
        else:
            checkpoint_controller.init_csv_logger(self.args, self.config)

        trainer.fit(dataloader_dict, length_dict, num_epochs=self.num_epochs, min_num_epochs=self.min_num_epochs,
                    save_model=True, parameter_controller=parameter_controller,
                    checkpoint_controller=checkpoint_controller)
