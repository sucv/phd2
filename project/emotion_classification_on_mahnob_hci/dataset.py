from base.dataset import VideoEmoRegressionArranger
import base.transforms3D as transforms3D
from base.utils import load_single_pkl

import os
from operator import itemgetter

import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class NFoldMahnobArranger(VideoEmoRegressionArranger):
    r"""
    A class to prepare files according to the n-fold manner.
    """

    def __init__(self, config, job, modality):

        # The root directory of the dataset.
        super().__init__(config)

        # Load the dataset information
        self.dataset_info = self.get_dataset_info()

        # Regression or Classification?
        self.job = job

        self.depth = config['window_length'] * config['continuous_label_frequency']
        self.step_size = config['hop_size'] * config['continuous_label_frequency']

        # Frame, EEG, which one or both?
        self.modality = modality
        self.get_modality_list()

        # Get the sessions having continuous labels.
        self.sessions_having_continuous_label = self.get_session_indices_having_continuous_label()

    def get_modality_list(self):

        # if self.job == "reg_v":
        self.modality.append("continuous_label")

    def get_session_indices_having_continuous_label(self):
        r"""
        Get the session indices having continuous labels.
        :return: (list), the indices indicating which sessions have continuous labels.
        """
        if self.job == "reg_v":
            indices = np.where(self.dataset_info['having_continuous_label'] == 1)[0]
        else:
            # indices = np.where(np.asarray(self.dataset_info['having_eeg']) == 1)[0]
            indices = np.where(self.dataset_info['having_continuous_label'] == 1)[0]
        return indices

    def make_data_dict(self, subject_id_of_all_folds, partition_dictionary):

        # Get the partition-wise subject dictionary.
        subject_id_of_all_partitions = self.partition_train_validate_test_for_subjects(
            subject_id_of_all_folds, partition_dictionary)

        # Initialize the dictionary to be outputed.
        data_dict = {key: [] for key in subject_id_of_all_partitions}

        for partition, subject_id_of_a_partition in subject_id_of_all_partitions.items():
            for subject_id_of_a_fold in subject_id_of_a_partition:

                data_of_a_modal = []
                for subject_id in subject_id_of_a_fold:

                    session_indices = self.get_session_index(subject_id)
                    if len(session_indices) > 1:
                        length_list = list(itemgetter(*session_indices)(self.dataset_info['refined_processed_length']))
                        session_name_list = list(itemgetter(*session_indices)(self.dataset_info['session_name']))
                        directory_list = [os.path.join(self.root_directory, self.npy_folder, session_name) for
                                          session_name in session_name_list]
                    else:
                        length_list = self.dataset_info['refined_processed_length'][session_indices[0]]
                        session_name_list = self.dataset_info['session_name'][session_indices[0]]
                        directory_list = os.path.join(self.root_directory, self.npy_folder, session_name_list)

                    initial_index_relative_to_this_subject = 0
                    for i in range(len(session_indices)):

                        if len(session_indices) > 1:
                            session_name = session_name_list[i]
                            length = length_list[i]
                            trial_directory = directory_list[i]
                        else:
                            session_name = session_name_list
                            length = length_list
                            trial_directory = directory_list

                        num_windows = int(np.ceil((length - self.depth) / self.step_size)) + 1

                        for window in range(num_windows):
                            start = window * self.step_size
                            end = start + self.depth

                            if end > length:
                                break

                            relative_indices = np.arange(start, end)
                            absolute_indices = relative_indices + initial_index_relative_to_this_subject
                            data_of_a_modal.append([trial_directory, absolute_indices, relative_indices, session_name])

                        if (length - self.depth) % self.step_size != 0:
                            start = length - self.depth
                            end = length
                            relative_indices = np.arange(start, end)
                            absolute_indices = relative_indices + initial_index_relative_to_this_subject
                            data_of_a_modal.append([trial_directory, absolute_indices, relative_indices, session_name])

                        if len(session_indices) > 1:
                            initial_index_relative_to_this_subject += length_list[i]

                data_dict[partition].extend(data_of_a_modal)

        return data_dict

    def make_length_dict(self, subject_id_of_all_folds, partition_dictionary):

        # Get the partition-wise subject dictionary.
        subject_id_of_all_partitions = self.partition_train_validate_test_for_subjects(
            subject_id_of_all_folds, partition_dictionary)

        # Initialize the dictionary to be outputed.
        length_dict = {key: {} for key in partition_dictionary}

        for partition, subject_id_of_a_partition in subject_id_of_all_partitions.items():
            for subject_id_of_a_fold in subject_id_of_a_partition:
                for subject_id in subject_id_of_a_fold:
                    indices = self.get_session_index(subject_id)
                    length_list = [self.dataset_info['refined_processed_length'][i] for i in indices]
                    length_dict_of_a_subject = {str(subject_id): length_list}
                    length_dict[partition] = {**length_dict[partition], **length_dict_of_a_subject}

        return length_dict

    def get_session_index(self, subject_id):
        indices = list(np.intersect1d(np.where(self.dataset_info['subject_id'] == subject_id)[0],
                                      self.sessions_having_continuous_label))
        return indices

    def get_subject_list_and_frequency(self):
        r"""
        Get the subject-wise session counts. It will be used for fold partition.
        :return: (list), the session counts of each subject.
        """
        subject_list, trial_count = np.unique(
            self.dataset_info['subject_id'][self.sessions_having_continuous_label], return_counts=True)
        return subject_list, trial_count

    def assign_session_to_subject(self):
        r"""
        Assign the sessions having continuous labels to its subjects.
        :return: (list), the list recording the session id having continuous labels for each subject.
        """
        subject_list, trial_count_for_each_subject = self.get_subject_list_and_frequency()

        session_id_of_each_subject = [
            np.where(self.dataset_info['subject_id'][self.sessions_having_continuous_label] == subject_id)[0] for
            _, subject_id in enumerate(subject_list)]
        return session_id_of_each_subject

    def assign_subject_to_fold(self, fold_number):
        r"""
        Assign the subjects and their sessions to a fold.
        :param fold_number: (int), how many fold the partition will create.
        :return: (list), the list recording the subject id and its associated session for each fold.
        """

        # Count the session number for each subject.
        subject_list, trial_count_for_each_subject = self.get_subject_list_and_frequency()

        # Calculate the expected session number for a fold, in order to partition it as evenly as possible.
        expected_trial_number_in_a_fold = np.sum(trial_count_for_each_subject) / fold_number

        # For preprocessing, or Leave One Subject Out scenario, which leaves one subject as a fold.
        if fold_number >= len(subject_list):
            expected_trial_number_in_a_fold = 0

        # In order to evenly partition the fold, we employ a simple algorithm. For each unprocessed
        # subjects, we always check if the current session number exceed the expected number. If
        # not, then assign the subject with the currently smallest number of session to the
        # current fold.

        # The mask is used to indicate whether the subject is assigned.
        mask = np.ones(len(subject_list), dtype=bool)

        subject_id_of_all_folds = []

        # Loop the subject.
        for i, (subject, trial_count) in enumerate(zip(subject_list, trial_count_for_each_subject)):

            # If the subject has not been assigned.
            if mask[i]:

                # Assign this subject to a new fold, then count the current session number,
                # and set the mask of this subject to False showing that it is assigned.
                one_fold = [subject]
                current_trial_number_in_a_fold = trial_count_for_each_subject[i]
                mask[i] = False

                # If the current session number is fewer than 90% of the expected number,
                # and there are still subjects that are not assigned.
                while (current_trial_number_in_a_fold <
                       expected_trial_number_in_a_fold * 0.9 and True in mask):
                    # Find the unassigned subject having the smallest session number currently.
                    trial_count_to_check = trial_count_for_each_subject[mask]
                    current_min_remaining = min(trial_count_to_check)

                    # Sometimes there are multiple subjects having the smallest number of session.
                    # If so, pick the first one to assign.
                    index_of_current_min = [j for j, count in
                                            enumerate(trial_count_for_each_subject)
                                            if (mask[j] and current_min_remaining == count)][0]

                    # Assign the subject to the fold.
                    one_fold.append(subject_list[index_of_current_min])

                    # Update the current count and mask.
                    current_trial_number_in_a_fold += current_min_remaining
                    mask[index_of_current_min] = False

                # Append the subjects of one fold to the final list.
                subject_id_of_all_folds.append(one_fold)

        # Also output the session id of all folds for convenience.
        session_id_of_all_folds = [np.hstack([np.where(
            self.dataset_info['subject_id'][self.sessions_having_continuous_label] == subject_id)[0]
                                              for subject_id in subject_id_of_one_fold])
                                   for subject_id_of_one_fold in subject_id_of_all_folds]

        return subject_id_of_all_folds, session_id_of_all_folds

    @staticmethod
    def partition_train_validate_test_for_subjects(subject_id_of_all_folds, partition_dictionary):
        r"""
        A static function to assign the subjects to folds according to the partition dictionary.
        :param subject_id_of_all_folds: (list), the list recording the subject id of all folds.
        :param partition_dictionary: (dict), the dictionary indicating the fold numbers of each partition.
        :return: (dict), the dictionary indicating the subject ids of each partition.
        """

        # To assure that the fold number equals the value sum of the partition dictionary.
        assert len(subject_id_of_all_folds) == np.sum([value for value in partition_dictionary.values()])

        # Assign the subject id according to the dictionary.
        subject_id_of_all_partitions = {
            'train': subject_id_of_all_folds[0:partition_dictionary['train']],
            'validate': subject_id_of_all_folds[partition_dictionary['train']:
                                                partition_dictionary['train'] + partition_dictionary['validate']],
            'test': subject_id_of_all_folds[-partition_dictionary['test']:]
        }

        return subject_id_of_all_partitions


class MAHNOBDataset(Dataset):
    def __init__(self, config, data_list, modality, time_delay=0, class_labels=None, mode='train'):
        self.config = config
        self.mode = mode
        self.data_list = data_list
        self.ratio = config['downsampling_interval_dict']
        self.modality = modality
        self.time_delay = np.int(time_delay * 4)
        self.get_3D_transforms()
        self.class_label = class_labels

    def get_3D_transforms(self):
        normalize = transforms3D.GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        normalize_eeg_image = transforms3D.GroupNormalize([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5])

        if self.mode == 'train':
            self.image_transforms = transforms.Compose([
                transforms3D.GroupNumpyToPILImage(0),
                transforms3D.GroupRandomCrop(self.config['frame_size'], self.config['crop_size']),
                transforms3D.GroupRandomHorizontalFlip(),
                transforms3D.Stack(),
                transforms3D.ToTorchFormatTensor(),
                normalize
            ])

            self.eeg_transforms = transforms.Compose([
                transforms3D.GroupWhiteNoiseByPCA(std_multiplier=0.1, num_components=2),
                transforms3D.ToTorchFormatTensor(),
                normalize_eeg_image
            ])

        else:
            self.image_transforms = transforms.Compose([
                transforms3D.GroupNumpyToPILImage(0),
                transforms3D.GroupCenterCrop(self.config['crop_size']),
                transforms3D.Stack(),
                transforms3D.ToTorchFormatTensor(),
                normalize
            ])

            self.eeg_transforms = transforms.Compose([
                transforms3D.ToTorchFormatTensor(),
                normalize_eeg_image
            ])

    def get_frame_indices(self, indices):
        x = 0
        if self.mode == 'train':
            x = random.randint(0, self.ratio['frame'] - 1)
        indices = indices * self.ratio['frame'] + x
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
        absolute_indices = self.data_list[index][1]
        relative_indices = self.data_list[index][2]
        session = self.data_list[index][3]
        new_indices = self.get_frame_indices(relative_indices)

        features = {}

        if "frame" in self.modality:
            frames = self.load_data(directory, new_indices, "frame.npy")
            frames = self.image_transforms(frames)
            features.update({'frame': frames})

        if "eeg_image" in self.modality:
            eeg_images = self.load_data(directory, relative_indices, "eeg_image.npy")
            eeg_images = self.eeg_transforms(eeg_images)
            features.update({'eeg_image': eeg_images})

        if self.class_label is None:
            # Regression on Valence
            labels = self.load_data(directory, relative_indices, "continuous_label.npy")

            # time delay
            labels = np.concatenate(
                (labels[self.time_delay:, :],
                 np.repeat(labels[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)
        else:
            labels = self.class_label[session]["Valence"]

        return features, labels, absolute_indices, session
