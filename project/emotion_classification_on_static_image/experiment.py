from base.experiment import GenericExperiment
from base.utils import init_weighted_sampler_and_weights
from base.loss_function import FocalLoss, CrossEntropyLoss
from project.emotion_classification_on_static_image.dataset import CKplusArranger, OuluArranger, RafdArranger, \
    EmotionalStaticImgClassificationDataset, FerplusCrossEntropyClassificationDataset
from base.checkpointer import ClassificationCheckpointer
from project.emotion_classification_on_static_image.trainer import Trainer
from project.emotion_classification_on_static_image.parameter_control import ParamControl
from models.model import my_res50

import os
from operator import itemgetter

import torch
from torch.utils import data

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomCrop, CenterCrop, RandomAffine, \
    ColorJitter, ToTensor, Normalize
import numpy as np


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)

        self.stamp = args.stamp
        self.num_folds = args.num_folds
        if args.folds_to_run is None:
            self.folds_to_run = np.arange(0, self.num_folds)

        self.cross_validation = args.cross_validation
        if not self.cross_validation:
            self.fold_to_run = [0]

        self.dataset_load_path = args.dataset_load_path
        self.save_model = args.save_model
        self.model_name = args.model_name + "_" + args.dataset + "_" + self.stamp
        self.model_mode = args.model_mode
        self.use_weighted_sampler = args.use_weighted_sampler

        self.learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        self.num_epochs = args.num_epochs
        self.min_num_epochs = args.min_num_epochs
        self.topk_accuracy = args.topk_accuracy
        self.batch_size = args.batch_size
        self.factor = args.factor
        self.patience = args.patience
        self.early_stopping = args.early_stopping

        self.milestone = [0]
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release

    def init_model(self):

        state_dict_name = self.config['state_dict_setting'][self.model_mode]
        model = my_res50(root_dir=self.model_load_path, num_classes=self.config['num_classes'], mode=self.model_mode,
                         use_pretrained=self.config['use_pretrained'], state_dict_name=state_dict_name)
        return model

    def load_config(self):
        if self.dataset == "affectnet":
            from project.emotion_classification_on_static_image.configs import config_affectnet as config

        elif self.dataset == "ck+":
            from project.emotion_classification_on_static_image.configs import config_ckplus as config

        elif self.dataset == "fer2013":
            from project.emotion_classification_on_static_image.configs import config_fer2013 as config

        elif self.dataset == "ferp" or self.dataset == "ferp_ce":
            from project.emotion_classification_on_static_image.configs import config_ferplus as config

        elif self.dataset == "rafd":
            from project.emotion_classification_on_static_image.configs import config_rafd as config

        elif self.dataset == "rafdb":
            from project.emotion_classification_on_static_image.configs import config_rafdb as config

        elif self.dataset == "oulu":
            from project.emotion_classification_on_static_image.configs import config_oulu as config
        else:
            raise ValueError("Unknown dataset!")

        return config

    def init_transform(self):

        transform = []
        transform.append(Resize(self.config['resize']))
        transform.append(RandomCrop(self.config['center_crop']))
        transform.append(ColorJitter())
        transform.append(RandomAffine(degrees=10))
        transform.append(RandomHorizontalFlip())
        transform.append(ToTensor())
        transform.append(Normalize(mean=self.config['mean'], std=self.config['std']))
        transform = Compose(transform)

        transform_val = []
        transform_val.append(Resize(self.config['resize']))
        transform_val.append(CenterCrop(self.config['center_crop']))
        transform_val.append(ToTensor())
        transform_val.append(Normalize(mean=self.config['mean'], std=self.config['std']))
        transform_val = Compose(transform_val)

        transform_dict = {'train': transform, 'validate': transform_val}
        return transform_dict

    def init_arranger(self):

        if self.dataset == "ckplus":
            arranger = CKplusArranger(self.config, self.num_folds)
        elif self.dataset == "oulu":
            arranger = OuluArranger(self.config, self.num_folds)
        elif self.dataset == "rafd":
            arranger = RafdArranger(self.config, self.num_folds)
        else:
            arranger = None

        return arranger

    def init_dataloader(self, transform_dict, fold=None, fold_list_origin=None):
        transform, transform_val = transform_dict['train'], transform_dict['validate']

        if self.cross_validation:
            fold_index = np.roll(self.folds_to_run, fold)
            fold_list = list(itemgetter(*fold_index)(fold_list_origin))

            # Need to specify according to the total fold number.
            # For example, for 10-fold, the partition is usually 7:2:1.
            train_list = np.vstack(fold_list[3:])
            validate_list = np.vstack(fold_list[1:3])
            test_list = np.vstack(fold_list[0])

            train_dataset = EmotionalStaticImgClassificationDataset(train_list, transform)
            validate_dataset = EmotionalStaticImgClassificationDataset(validate_list, transform_val)
            test_dataset = EmotionalStaticImgClassificationDataset(test_list, transform)

        else:

            if self.dataset == "ferp_ce":
                train_dataset = FerplusCrossEntropyClassificationDataset(self.dataset_load_path, mode='train',
                                                                         transform=transform)
                validate_dataset = FerplusCrossEntropyClassificationDataset(self.dataset_load_path, mode='validate',
                                                                            transform=transform_val)
                test_dataset = FerplusCrossEntropyClassificationDataset(self.dataset_load_path, mode='test',
                                                                        transform=transform_val)
            else:
                train_dataset = ImageFolder(os.path.join(self.dataset_load_path, 'train'),
                                            transform=transform)

                validate_dataset = ImageFolder(os.path.join(self.dataset_load_path, 'validate'),
                                               transform=transform_val)
                # Some datasets did not provide test sets.
                test_dataset = None
                if os.path.isdir(os.path.join(self.dataset_load_path, 'test')):
                    test_dataset = ImageFolder(os.path.join(self.dataset_load_path, 'test'),
                                               transform=transform_val)

        sampler, sample_weights = None, None
        if self.use_weighted_sampler and not self.dataset == "ferp_ce":
            sampler, sample_weights = init_weighted_sampler_and_weights(train_dataset)

        if sampler is None:
            train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, sampler=sampler)

        validate_loader = data.DataLoader(dataset=validate_dataset, batch_size=self.batch_size)
        test_loader = None
        if test_dataset is not None:
            test_loader = data.DataLoader(dataset=test_dataset, batch_size=self.batch_size)

        dataloader_dict = {'train': train_loader, 'validate': validate_loader, 'test': test_loader}
        return dataloader_dict, sample_weights

    def experiment(self):
        device = self.init_device()
        fold_list_origin = None

        save_path = os.path.join(self.model_save_path, self.model_name)

        if self.cross_validation:
            arranger = self.init_arranger()
            fold_list_origin = arranger.establish_fold()

        transform_dict = self.init_transform()
        criterion = FocalLoss()
        if self.dataset == "ferp_ce":
            criterion = CrossEntropyLoss()

        for fold in iter(self.fold_to_run):
            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)

            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            dataloader_dict, samples_weights = self.init_dataloader(transform_dict=transform_dict, fold=fold,
                                                                    fold_list_origin=fold_list_origin)
            model = self.init_model()

            milestone = self.milestone

            trainer = Trainer(model, model_name=self.model_name, save_path=fold_save_path,
                              criterion=criterion, num_classes=self.config['num_classes'],
                              device=device, factor=self.factor,
                              learning_rate=self.learning_rate, fold=fold, milestone=milestone,
                              patience=self.patience, early_stopping=self.early_stopping,
                              min_learning_rate=self.min_learning_rate,
                              samples_weight=samples_weights)

            parameter_controller = ParamControl(trainer, gradual_release=self.gradual_release,
                                                release_count=self.release_count, backbone_mode=self.model_mode)
            checkpoint_controller = ClassificationCheckpointer(checkpoint_filename, trainer, parameter_controller,
                                                               resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            trainer.fit(dataloader_dict, num_epochs=self.num_epochs, topk_accuracy=self.topk_accuracy,
                        min_num_epochs=self.min_num_epochs, save_model=self.save_model,
                        parameter_controller=parameter_controller, checkpoint_controller=checkpoint_controller)

            if dataloader_dict['test'] is not None:
                trainer.test(data_to_load=dataloader_dict, topk_accuracy=1, checkpoint_controller=checkpoint_controller)
                checkpoint_controller.save_checkpoint(trainer, parameter_controller, fold_save_path)
