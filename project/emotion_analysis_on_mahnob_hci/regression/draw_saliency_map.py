from base.experiment import GenericExperiment
from models.model import my_2d1d, my_2dlstm, my_eeg1d, my_temporal, my_eeglstm, soley_lstm
from base.dataset import NFoldMahnobArrangerLOSO, MAHNOBDatasetTrial
from project.emotion_analysis_on_mahnob_hci.regression.checkpointer import Checkpointer
from project.emotion_analysis_on_mahnob_hci.regression.trainer import MAHNOBRegressionTrainerTrial
from project.emotion_analysis_on_mahnob_hci.regression.parameter_control import ParamControl
from base.utils import load_single_pkl
from base.loss_function import CCCLoss

import os
from operator import itemgetter

import random
import numpy as np
import torch
import torch.nn
from torch.utils import data

import mne
import matplotlib.pyplot as plt


def load_channel_name():
    ch_name = []
    with open("channel_name.txt", "r") as f:
        ch_name = [l.strip() for l in f]
    return ch_name


def plot_topo(data, index):
    biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    n_channels = len(biosemi_montage.ch_names)
    ch_names = load_channel_name()
    fake_info = mne.create_info(ch_names=ch_names, sfreq=250.,
                                ch_types='eeg')
    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(biosemi_montage)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), gridspec_kw=dict(top=1.0),
                           sharex=True, sharey=True)

    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    im, _ = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax,
                         show=True,  names=ch_names, show_names=True,)
    #fig.colorbar(im, cax=ax, orientation='vertical')
    fig.savefig('pic_{}.pdf'.format(index))

def normalize(data):
    data = np.array(data)
    max_v = np.max(data)
    min_v = np.min(data)
    data = (data - min_v) / (max_v - min_v)
    return data


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)

        self.num_folds = 24
        self.folds_to_run = args.folds_to_run
        self.include_session_having_no_continuous_label = 0

        self.stamp = args.stamp
        self.case = args.case

        self.modality = args.modality
        self.model_name = self.experiment_name + "_" + "1d" + "_" + "reg_v" + "_" + self.modality[0] + "_" + self.case + "_" + self.stamp
        self.backbone_state_dict_frame = args.backbone_state_dict_frame
        self.backbone_state_dict_eeg = args.backbone_state_dict_eeg
        self.backbone_mode = args.backbone_mode

        self.cnn1d_embedding_dim = 192
        self.cnn1d_channels = [128, 128]
        self.cnn1d_kernel_size = 3
        self.cnn1d_dropout = args.cnn1d_dropout
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout

        self.psd_num_inputs = args.psd_num_inputs

        self.window_sec = 250
        self.hop_size_sec = 250
        self.continuous_label_frequency = args.continuous_label_frequency
        self.frame_size = args.frame_size
        self.crop_size = args.crop_size
        self.batch_size = args.batch_size

        self.num_classes = args.num_classes
        self.emotion_dimension = args.emotion_dimension

        self.device = self.init_device()


    def load_config(self):
        from project.emotion_analysis_on_mahnob_hci.configs import config_mahnob as config
        return config

    def create_model(self):

        self.init_random_seed()

        output_dim = 1
        model = my_temporal(model_name=self.model_name, num_inputs=self.psd_num_inputs, cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                            cnn1d_dropout_rate=self.cnn1d_dropout, embedding_dim=self.lstm_embedding_dim, hidden_dim=self.lstm_hidden_dim, lstm_dropout_rate=self.lstm_dropout, output_dim=output_dim)

        return model

    def init_partition_dictionary(self):
        partition_dictionary = {'train': 23, 'validate': 0, 'test': 1}
        return partition_dictionary

    def combine_trial_for_partition(self, subject_id_of_all_folds, trial_id_to_subject_dict):
        subject_id_of_non_test_subjects = [subject[0] for subject in subject_id_of_all_folds[:-1]]
        subject_id_of_the_test_subject = subject_id_of_all_folds[-1]

        trial_id_of_non_test_subjects, trial_id_of_the_test_subject = [], []
        [trial_id_of_non_test_subjects.extend(trial_id_to_subject_dict[subject]) for subject in subject_id_of_non_test_subjects]
        [trial_id_of_the_test_subject.extend(trial_id_to_subject_dict[subject]) for subject in subject_id_of_the_test_subject]

        random.shuffle(trial_id_of_non_test_subjects)

        train_validate_length = len(trial_id_of_non_test_subjects)

        train_length = int(train_validate_length * 0.8)

        trial_id_of_all_partitions = {'train': trial_id_of_non_test_subjects[:train_length], 'validate': trial_id_of_non_test_subjects[train_length:], 'test': trial_id_of_the_test_subject}

        return trial_id_of_all_partitions

    def init_dataloader(self, subject_id_of_all_folds, trial_id_to_subject_dict, fold_arranger, fold, class_labels=None):

        # Set the fold-to-partition configuration.
        # Each fold have approximately the same number of sessions.
        self.init_random_seed()
        partition_dictionary = self.init_partition_dictionary()

        fold_index = np.roll(np.arange(len(subject_id_of_all_folds)), fold)
        subject_id_of_all_folds = list(itemgetter(*fold_index)(subject_id_of_all_folds))
        trial_id_of_all_partitions = self.combine_trial_for_partition(subject_id_of_all_folds, trial_id_to_subject_dict)
        data_dict, normalize_dict = fold_arranger.make_data_dict(trial_id_of_all_partitions)
        print(subject_id_of_all_folds)

        dataloaders_dict = {}
        for partition in partition_dictionary.keys():
            dataset = MAHNOBDatasetTrial(self.config, data_dict[partition], normalize_dict=normalize_dict,
                                         modality=self.modality,
                                         continuous_label_frequency=self.config['frequency_dict']['continuous_label'],
                                         normalize_eeg_raw=0,
                                         emotion_dimension=self.emotion_dimension,
                                         eegnet_window_sec=0,
                                         eegnet_stride_sec=0,
                                         time_delay=0, class_labels=class_labels, mode=partition)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=True if partition == "train" else False)

        return dataloaders_dict, normalize_dict

    def experiment(self):

        save_path = os.path.join(self.model_save_path, self.model_name)

        fold_arranger = NFoldMahnobArrangerLOSO(dataset_load_path=self.dataset_load_path, normalize_eeg_raw=0,
                                            dataset_folder=self.dataset_folder, window_sec=self.window_sec, hop_size_sec=self.hop_size_sec,
                                            include_session_having_no_continuous_label=self.include_session_having_no_continuous_label,
                                            modality=self.modality, feature_extraction=False)
        subject_id_of_all_folds, trial_id_to_subject_dict = fold_arranger.assign_subject_to_fold(self.num_folds)
        print(subject_id_of_all_folds)

        subjectwise_saliency_dict = {}
        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.num_folds))

            model = self.create_model()

            fold_save_path = os.path.join(save_path, str(fold))
            os.makedirs(fold_save_path, exist_ok=True)



            dataloaders_dict, normalize_dict = self.init_dataloader(subject_id_of_all_folds, trial_id_to_subject_dict, fold_arranger, fold)

            model_path = os.path.join("/home/zhangsu/phd2/save/group_kd_final7_1d_only_reg_v_eeg_psd_loso_128x2_lr1e5_ccc_weight_1.0_kd_weight_0_l1_1", str(fold), "model_state_dict.pth")
            model_state = torch.load(model_path)
            model.load_state_dict(model_state)
            # model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False

            trialwise_saliancy_dict = {}
            for trial_path, indices,  length, trial_name in dataloaders_dict['test'].dataset.data_list:
                subject = trial_name.split("P")[1].split("-")[0]
                trial = trial_name.split("T")[1]
                if subject not in subjectwise_saliency_dict:
                    subjectwise_saliency_dict[subject] = {}
                trial_path = os.path.join(trial_path, "eeg_psd.npy")
                indices = indices[:-length]
                data = np.load(trial_path)
                data = data[np.newaxis, :]
                eeg_psd = torch.from_numpy(data).float()
                eeg_psd = eeg_psd.transpose(1, 2).contiguous()
                eeg_psd.requires_grad_()
                eeg_psd.retain_grad()
                eeg_psd.cuda(0)
                model.eval()
                # eeg_psd.register_hook(lambda x: print(x))
                output, _ = model(eeg_psd)
                max_index = output.argmax()
                min_index = output.argmin()
                output_max = output[0, max_index, 0]
                output_min = output[0, min_index, 0]
                output_max.backward()

                saliency = [normalize(torch.squeeze(torch.mean(eeg_psd.grad.data.abs()[:,i::6,:], axis=2))) for i in range(6)]
                saliency = np.stack(saliency)

                subjectwise_saliency_dict[subject][trial] = saliency

        for subject, subjects_trials in subjectwise_saliency_dict.items():
            cache = np.zeros((6, 32))
            count = 0
            for trial, trials_bands in subjects_trials.items():
                cache += trials_bands
                count += 1
            cache = cache / count
            subjectwise_saliency_dict[subject] = cache

            mean = np.mean(cache[0])
            std = np.std(cache[0])
            cache[0] = (cache[0] - mean) / std

            # max_v = np.max(data)
            # min_v = np.min(data)
            # data = (data - min_v)/(max_v - min_v)

            # data = np.stack((data, data), axis=-1)
            plot_topo(cache[0], int(subject))

        print(0)





