from base.experiment import GenericExperiment
from base.utils import init_weighted_sampler_and_weights
from base.loss_function import FocalLoss
from project.emotion_classification_on_static_image.dataset import CKplusArranger, OuluArranger, RafdArranger, \
    EmotionalStaticImgClassificationDataset
from project.emotion_classification_on_static_image.checkpointer import Checkpointer
from project.emotion_classification_on_static_image.trainer import ImageClassificationTrainer
from project.emotion_classification_on_static_image.parameter_control import ParamControl
from models.model import my_res50

import os
from operator import itemgetter

import torch
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomCrop, CenterCrop, RandomAffine, ColorJitter, ToTensor, Normalize
import numpy as np


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.stamp = args.s
        self.num_folds = args.n_fold
        if args.fold_to_run is None:
            self.fold_to_run = np.arange(0, self.num_folds)

        if not args.cv:
            self.fold_to_run = [0]

        self.dataset = args.d
        self.model = args.m
        self.cross_validation = args.cv
        self.model_name = args.m + "_" + args.d

    def init_model(self):
        model = my_res50(root_dir=self.model_load_path, num_classes=self.config['num_classes'], mode="ir",
                         use_pretrained=self.config['use_pretrained'], state_dict_name="backbone_ir50_ms1m_epoch120")
        return model

    def load_config(self):
        if self.args.d == "affectnet":
            from project.emotion_classification_on_static_image.configs import config_affectnet as config

        elif self.args.d == "ck+":
            from project.emotion_classification_on_static_image.configs import config_ckplus as config

        elif self.args.d == "fer2013":
            from project.emotion_classification_on_static_image.configs import config_fer2013 as config

        elif self.args.d == "fer+":
            from project.emotion_classification_on_static_image.configs import config_ferplus as config

        elif self.args.d == "rafd":
            from project.emotion_classification_on_static_image.configs import config_rafd as config

        elif self.args.d == "rafdb":
            from project.emotion_classification_on_static_image.configs import config_rafdb as config

        elif self.args.d == "oulu":
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
            fold_index = np.roll(self.fold_to_run, fold)
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
            train_dataset = ImageFolder(os.path.join(self.config['remote_root_directory'], 'train'),
                                        transform=transform)

            validate_dataset = ImageFolder(os.path.join(self.config['remote_root_directory'], 'validate'),
                                      transform=transform_val)
            # Some datasets did not provide test sets.

            test_dataset = None
            if os.path.isdir(os.path.join(self.config['remote_root_directory'], 'test')):
                test_dataset = ImageFolder(os.path.join(self.config['remote_root_directory'], 'test'),
                                           transform=transform_val)

        sampler, sample_weights = init_weighted_sampler_and_weights(train_dataset)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], sampler=sampler)
        validate_loader = data.DataLoader(dataset=validate_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = None
        if test_dataset is not None:
            test_loader = data.DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'], shuffle=False)

        dataloader_dict = {'train': train_loader, 'validate': validate_loader, 'test': test_loader}
        return dataloader_dict, sample_weights

    def experiment(self):
        device = self.init_device()
        fold_list_origin = None

        save_path = os.path.join(self.model_save_path, self.model_name + "_" + self.stamp)

        if self.cross_validation:
            arranger = self.init_arranger()
            fold_list_origin = arranger.establish_fold()

        transform_dict = self.init_transform()

        for fold in iter(self.fold_to_run):
            fold_save_path = os.path.join(save_path, str(fold))
            checkpoint_filename = os.path.join(fold_save_path, "checkpoint.pkl")

            dataloader_dict, samples_weights = self.init_dataloader(transform_dict=transform_dict, fold=fold,
                                                                    fold_list_origin=fold_list_origin)
            model = self.init_model()
            criterion = FocalLoss()
            milestone = [0]

            trainer = ImageClassificationTrainer(model, model_name=self.model_name, save_path=fold_save_path, criterion=criterion,
                                                 num_classes=self.config['num_classes'], device=device, learning_rate=1e-3,
                                                 fold=fold, milestone=milestone, patience=20, early_stopping=100, min_learning_rate=1e-5,
                                                 samples_weight=samples_weights)

            parameter_controller = ParamControl(trainer, release_count=3)
            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger()

            trainer.fit(dataloader_dict, num_epochs=500, topk_accuracy=1, min_num_epoch=0, save_model=True,
                        parameter_controller=parameter_controller, checkpoint_controller=checkpoint_controller)

            # path = os.path.join(directory_to_save_trained_model_and_csv, "state_dict" + ".pth")
            # state_dict = torch.load(path, map_location='cpu')
            # model.load_state_dict(state_dict['net_state_dict'])
            # test_loss, test_acc = trainer.validate(dataloader_dict['test'], topk_accuracy=1)
            # print("The test accuracy on fold {} is {}".format(str(fold), str(test_acc)))
