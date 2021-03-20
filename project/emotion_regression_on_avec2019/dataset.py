from base.dataset import VideoEmoRegressionArranger
import base.transforms3D as transforms3D
import os

import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class AVEC2019Arranger(VideoEmoRegressionArranger):
    def __init__(self, config):
        super().__init__(config)

    def make_length_dict(self, train_country="all", validate_country="all"):
        length_dict = self.generate_length_dict()
        if train_country == "all":
            train_dict = {**length_dict['Train']['DE'], **length_dict['Train']['HU']}
        else:
            train_dict = length_dict['Train'][train_country]

        if validate_country == "all":
            validate_dict = {**length_dict['Devel']['DE'], **length_dict['Devel']['HU']}
        else:
            validate_dict = length_dict['Devel'][validate_country]

        length_dict = {'train': train_dict, 'validate': validate_dict}
        return length_dict

    def generate_length_dict(self):
        length_dict = self.init_length_dict()

        for index, subject_id in enumerate(self.dataset_info['subject_id']):
            length = [self.dataset_info['frame_count'][index] // 5]
            country = self.dataset_info['country'][index]
            partition = self.dataset_info['partition'][index]
            length_dict[partition][country].update({str(subject_id): length})

        return length_dict

    def make_data_dict(self, train_country="all", validate_country="all"):
        data_dict = self.generate_data_dict()
        if train_country == "all":
            train_dict = data_dict['Train']['DE'] + data_dict['Train']['HU']
        else:
            train_dict = data_dict['Train'][train_country]

        if validate_country == "all":
            validate_dict = data_dict['Devel']['DE'] + data_dict['Devel']['HU']
        else:
            validate_dict = data_dict['Devel'][validate_country]

        data_dict = {'train': train_dict, 'validate': validate_dict}
        return data_dict

    def generate_data_dict(self):
        data_dict = self.init_data_dict()
        directory = os.path.join(self.root_directory, self.npy_folder)
        for index, trial in enumerate(sorted(os.listdir(directory))):

            trial_directory = os.path.join(directory, trial)
            country = self.get_country(self.dataset_info['partition_country_to_participant_trial_map'][trial])
            partition = self.get_partition(self.dataset_info['partition_country_to_participant_trial_map'][trial])
            session = trial_directory.split(os.sep)[-1]
            length = self.dataset_info['frame_count'][index] // 5

            item_list = []

            # sampler_length = self.window_length
            # if partition == "Devel":
            #     sampler_length = length

            # if partition == "Train":
            sampler_length = self.window_length
            start = 0
            end = start + sampler_length

            while end <= length:
                indices = np.arange(start, end)
                item_list.append([trial_directory, indices, session])
                start = start + self.hop_size
                end = start + sampler_length

            if end > length:
                end = length
                start = length - sampler_length
                indices = np.arange(start, end)
                item_list.append([trial_directory, indices, session])

            # else:
            #     start = 0
            #     end = length
            #     indices = np.arange(start, end)
            #     item_list.append([trial_directory, indices, session])

            data_dict[partition][country].extend(item_list)

            # for window in range(num_windows):
            #     start = window * self.hop_size
            #     end = start + self.window_length
            #
            #     if end > length:
            #         break
            #
            #     indices = np.arange(start, end)
            #     item_list.append([trial_directory, indices, session])
            #
            # if (length - self.window_length) % self.hop_size != 0:
            #     start = length - self.window_length
            #     end = length
            #     indices = np.arange(start, end)
            #     item_list.append([trial_directory, indices, session])

            # ==================================
            # if partition == 'Train':
            #     for window in range(num_windows):
            #         start = window * self.hop_size
            #         end = start + self.window_length
            #         indices = np.arange(start, end)
            #         item_list.append([trial_directory, indices, session])
            #
            #     if (length - self.window_length) % self.hop_size != 0:
            #         start = length - self.window_length
            #         end = length
            #         indices = np.arange(start, end)
            #         item_list.append([trial_directory, indices, session])
            #
            # elif partition == 'Devel':
            #     status = 0
            #     indices = np.arange(0, length)
            #     item_list.append([trial_directory, indices, session])
            # else:
            #     raise ValueError('Partition not supported!')
            #
            # data_dict[partition][country].extend(item_list)

        return data_dict

    @staticmethod
    def init_length_dict():
        length_dict = {"Train": {
            "DE": {},
            "HU": {},
        },
            "Devel": {
                "DE": {},
                "HU": {},
            }}
        return length_dict

    @staticmethod
    def init_data_dict():
        data_dict = {"Train": {
            "DE": [],
            "HU": [],
        },
            "Devel": {
                "DE": [],
                "HU": [],
            }}
        return data_dict

    @staticmethod
    def get_partition(string):
        partition = string.split("_")[0]
        return partition

    @staticmethod
    def get_country(string):
        country = string.split("_")[1]
        return country


class AVEC2019Dataset(Dataset):
    def __init__(self, config, data_list, time_delay=0, emotion="a", head="multi-headed", mode='train'):
        self.config = config
        self.mode = mode
        self.data_list = data_list
        self.frame_to_label_ratio = config['downsampling_interval_dict']['frame']
        self.time_delay = np.int(time_delay * 10)
        self.emotion = emotion
        self.head = head
        self.get_3D_transforms()

    def get_frame_indices(self, indices):
        x = 0
        if self.mode == 'train':
            x = random.randint(0, self.frame_to_label_ratio - 1)
        indices = indices * self.frame_to_label_ratio + x
        return indices

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def load_data(directory, indices, filename):
        filename = os.path.join(directory, filename)
        frames = np.load(filename, mmap_mode='c')[indices]
        return frames

    def __getitem__(self, index):

        directory = self.data_list[index][0]
        indices = self.data_list[index][1]
        session = self.data_list[index][2]
        new_indices = self.get_frame_indices(indices)

        frames = self.load_data(directory, new_indices, "frame.npy")
        frames = self.transforms(frames)

        continuous_labels = self.load_data(directory, indices, "continuous_label.npy")

        if self.head == "single-headed":
            if self.emotion == "arousal": #Arousal
                continuous_labels = continuous_labels[:, 0][:, np.newaxis]
            elif self.emotion == "valence": # Valence
                continuous_labels = continuous_labels[:, 1][:, np.newaxis]
            else:
                raise ValueError("Unsupported emotional dimension for continuous labels!")

        # time delay
        continuous_labels = np.concatenate((continuous_labels[self.time_delay:, :],
                        np.repeat(continuous_labels[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        return frames, continuous_labels, indices, session

    def get_3D_transforms(self):
        normalize = transforms3D.GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms3D.GroupNumpyToPILImage(0),
                transforms3D.GroupCenterCrop(self.config['crop_size']),
                transforms3D.GroupRandomHorizontalFlip(),
                transforms3D.Stack(),
                transforms3D.ToTorchFormatTensor(),
                normalize
            ])
        else:
            self.transforms = transforms.Compose([
                transforms3D.GroupNumpyToPILImage(0),
                transforms3D.GroupCenterCrop(self.config['crop_size']),
                transforms3D.Stack(),
                transforms3D.ToTorchFormatTensor(),
                normalize
            ])