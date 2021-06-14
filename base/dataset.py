import base.transforms3D as transforms3D
from base.utils import load_single_pkl

import os
from operator import itemgetter

import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class VideoEmoRegressionArranger(object):
    def __init__(self, dataset_load_path, dataset_folder):
        self.root_directory = dataset_load_path
        self.npy_folder = dataset_folder
        self.dataset_info = self.get_dataset_info()

    def make_length_dict(self, **kwargs):
        pass

    def make_data_dict(self, **kwargs):
        pass

    def get_dataset_info(self):
        r"""
        Read the dataset info pkl file.
        :return: (dict), the dataset info.
        """
        dataset_info = load_single_pkl(self.root_directory, "dataset_info")
        return dataset_info


class ImageEmoClassificationNFoldArranger(object):
    def __init__(self, config, num_folds):
        self.root_directory = config['remote_root_directory']
        self.num_folds = num_folds
        self.emotion_dict = self.init_emotion_dict()
        self.subject_list = self.get_subject_list()

    @staticmethod
    def init_emotion_dict():
        raise NotImplementedError

    def establish_fold(self):
        fold_list = [[] for _ in range(self.num_folds)]
        return fold_list

    def get_subject_list(self):
        subject_list = [folder for folder in os.listdir(self.root_directory)]
        return subject_list

    @staticmethod
    def count_subject_for_each_fold():
        raise NotImplementedError


class NFoldMahnobArrangerTrial(VideoEmoRegressionArranger):
    r"""
    A class to prepare files according to the n-fold manner. All the trials are shuffled first, and then evenly separate to n-fold.
        Each fold contains trials from 1 or more subjects.  Subjects CAN overlap among folds.
    """

    def __init__(self, dataset_load_path, dataset_folder, modality, normalize_eeg_raw=True, num_electrodes=32, window_sec=24, hop_size_sec=8, continuous_label_frequency=4,
                 partition_setting=None, include_session_having_no_continuous_label=True, feature_extraction=False):

        # The root directory of the dataset.
        super().__init__(dataset_load_path=dataset_load_path, dataset_folder=dataset_folder)

        # Load the dataset information

        self.dataset_info = self.get_dataset_info()

        self.partition_setting = partition_setting
        self.num_electrodes = num_electrodes
        self.normalize_eeg_raw = normalize_eeg_raw

        # Exclude trials having no continuous label.
        self.include_session_having_no_continuous_label = include_session_having_no_continuous_label

        self.depth = window_sec * continuous_label_frequency
        self.step_size = hop_size_sec * continuous_label_frequency

        # Frame, EEG, which one or both?
        self.modality = modality
        self.get_modality_list()

        self.feature_extraction = feature_extraction

    def get_modality_list(self):

        # if self.job == "reg_v":
        self.modality.append("continuous_label")

    def get_trial_indices_having_continuous_label(self):
        r"""
        Get the session indices having continuous labels.
        :return: (list), the indices indicating which sessions have continuous labels.
        """

        # If include trial
        if self.include_session_having_no_continuous_label:
            indices = np.where(np.asarray(self.dataset_info['having_eeg']) == 1)[0]
        else:
            indices = np.where(self.dataset_info['having_continuous_label'] == 1)[0]

        np.random.shuffle(indices)
        return indices

    def make_data_dict(self, trial_id_of_all_partitions):

        # Initialize the dictionary to be outputed.
        data_dict = {key: [] for key in trial_id_of_all_partitions}
        length_dict = {key: [] for key in trial_id_of_all_partitions}
        normalize_dict = {key: {} for key in trial_id_of_all_partitions}

        for partition, trial_id_of_a_partition in trial_id_of_all_partitions.items():

            data_of_a_partition = []
            length_of_a_partition = []
            if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                trial_wise_eeg_mean_cache = np.zeros((self.num_electrodes,))
                trial_wise_eeg_std_cache = np.zeros((self.num_electrodes,))
                num_samples = 0
                eeg_raw_path_list = []

            trial_indices = trial_id_of_a_partition

            if len(trial_indices) == 1:
                length_list = [self.dataset_info['refined_processed_length'][int([trial_indices][0])]]
                trial_name_list = [self.dataset_info['session_name'][int([trial_indices][0])]]
            else:
                length_list = list(itemgetter(*trial_indices)(self.dataset_info['refined_processed_length']))
                trial_name_list = list(itemgetter(*trial_indices)(self.dataset_info['session_name']))
            directory_list = [os.path.join(self.root_directory, self.npy_folder, session_name) for
                                  session_name in trial_name_list]

            for i in range(len(trial_indices)):
                trial_name = trial_name_list[i]
                length = length_list[i]
                trial_directory = directory_list[i]


                if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                    eeg_raw_path = os.path.join(trial_directory, "eeg_raw.npy")
                    intermediate = np.load(eeg_raw_path)
                    num_samples += intermediate.shape[0]
                    trial_wise_eeg_mean_cache += np.sum(intermediate, axis=0)
                    eeg_raw_path_list.append(eeg_raw_path)

                if self.feature_extraction:
                    depth = length
                else:
                    depth = self.depth

                start = 0
                end = start + depth

                while end <= length:
                    relative_indices = np.arange(start, end)
                    data_of_a_partition.append([trial_directory, relative_indices, length, trial_name])
                    start += self.step_size
                    end = start + depth

                if (length - depth) % self.step_size != 0:
                    start = length - depth
                    end = length
                    relative_indices = np.arange(start, end)
                    data_of_a_partition.append([trial_directory, relative_indices, length, trial_name])

            data_dict[partition].extend(data_of_a_partition)

            if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                eeg_raw_mean = trial_wise_eeg_mean_cache / num_samples

                for eeg_raw_path in eeg_raw_path_list:
                    trial_wise_eeg_std_cache += np.sum(np.square(np.load(eeg_raw_path) - eeg_raw_mean), axis=0)

                eeg_raw_std = np.sqrt(trial_wise_eeg_std_cache / num_samples)
                normalize_dict[partition]['mean'] = eeg_raw_mean
                normalize_dict[partition]['std'] = eeg_raw_std

        return data_dict, normalize_dict

    def assign_trial_to_partition(self, trial_indices):
        partition_dict = {}

        partition_dict['train'] = trial_indices[:self.partition_setting['train']]
        if not self.feature_extraction:
            partition_dict['validate'] = trial_indices[self.partition_setting['train']:self.partition_setting['train']+self.partition_setting['validate']]
            partition_dict['test'] = trial_indices[-self.partition_setting['test']:]
        return partition_dict

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


class NFoldMahnobArranger(VideoEmoRegressionArranger):
    r"""
    A class to prepare files according to the n-fold manner. A fold contains trials from 1 or more subjects.
        No subject overlap among folds.
    """

    def __init__(self, dataset_load_path, dataset_folder, modality, normalize_eeg_raw=True, num_electrodes=32, window_sec=24, hop_size_sec=8, continuous_label_frequency=4, include_session_having_no_continuous_label=True, feature_extraction=False):

        # The root directory of the dataset.
        super().__init__(dataset_load_path=dataset_load_path, dataset_folder=dataset_folder)

        # Load the dataset information
        self.dataset_info = self.get_dataset_info()

        self.num_electrodes = num_electrodes
        self.normalize_eeg_raw = normalize_eeg_raw

        # Regression or Classification?
        self.include_session_having_no_continuous_label = include_session_having_no_continuous_label

        self.depth = window_sec * continuous_label_frequency
        self.step_size = hop_size_sec * continuous_label_frequency

        # Frame, EEG, which one or both?
        self.modality = modality
        self.get_modality_list()

        # Get the sessions having continuous labels.
        self.sessions_having_continuous_label = self.get_session_indices_having_continuous_label()
        self.feature_extraction = feature_extraction

    def get_modality_list(self):

        # if self.job == "reg_v":
        self.modality.append("continuous_label")

    def get_session_indices_having_continuous_label(self):
        r"""
        Get the session indices having continuous labels.
        :return: (list), the indices indicating which sessions have continuous labels.
        """
        if self.include_session_having_no_continuous_label:
            indices = np.where(np.asarray(self.dataset_info['having_eeg']) == 1)[0]
        else:
            indices = np.where(self.dataset_info['having_continuous_label'] == 1)[0]
        return indices

    def make_data_dict(self, subject_id_of_all_folds, partition_dictionary):
        r"""
        To generate the dictionary containing path, window indices, trial names, and so on.
            Each element in the dictionary is a data point, it tells the Dataset class where and what to load the data.
        """

        # Get the partition-wise subject dictionary.
        subject_id_of_all_partitions = self.partition_train_validate_test_for_subjects(
            subject_id_of_all_folds, partition_dictionary)

        # Initialize the dictionary to be outputed.
        data_dict = {key: [] for key in subject_id_of_all_partitions}

        normalize_dict = {key: {} for key in subject_id_of_all_partitions}

        for partition, subject_id_of_a_partition in subject_id_of_all_partitions.items():

            if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                trial_wise_eeg_mean_cache = np.zeros((self.num_electrodes,))
                trial_wise_eeg_std_cache = np.zeros((self.num_electrodes,))
                num_samples = 0
                eeg_raw_path_list = []

            for subject_id_of_a_fold in subject_id_of_a_partition:

                for subject_id in subject_id_of_a_fold:
                    data_of_a_modal = []
                    session_indices = self.get_session_index(subject_id)
                    if len(session_indices) > 0:
                        if len(session_indices) > 1:
                            length_list = list(itemgetter(*session_indices)(self.dataset_info['refined_processed_length']))
                            session_name_list = list(itemgetter(*session_indices)(self.dataset_info['session_name']))
                            directory_list = [os.path.join(self.root_directory, self.npy_folder, session_name) for
                                              session_name in session_name_list]

                        # If found only one session
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

                            if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                                eeg_raw_path = os.path.join(trial_directory, "eeg_raw.npy")
                                intermediate = np.load(eeg_raw_path)
                                num_samples += intermediate.shape[0]
                                trial_wise_eeg_mean_cache += np.sum(intermediate, axis=0)
                                eeg_raw_path_list.append(eeg_raw_path)

                            if self.feature_extraction:
                                # During which, a whole trial is fed to the model for convenience, so that the depth
                                # equals the trial length.
                                depth = length
                            else:
                                # Otherwise, load data according to the window size.
                                depth = self.depth
                            num_windows = int(np.ceil((length - depth) / self.step_size)) + 1

                            for window in range(num_windows):
                                start = window * self.step_size
                                end = start + depth

                                if end > length:
                                    break

                                # Relative and absolute indices are used for:
                                #   1) plot the trial-wise output-label trace figures.
                                #   2) calculate the evaluation metrics on the output-label pair.
                                #   To do so, the window-resampled output must be restored to a complete trial (for plotting),
                                #       and then all the output of trials will be concatenated to be one single vector as the partition output (for evaluation).
                                relative_indices = np.arange(start, end)
                                absolute_indices = relative_indices + initial_index_relative_to_this_subject
                                data_of_a_modal.append([trial_directory, absolute_indices, relative_indices, session_name])

                            # Make sure no data are omitted.
                            if (length - depth) % self.step_size != 0:
                                start = length - depth
                                end = length
                                relative_indices = np.arange(start, end)
                                absolute_indices = relative_indices + initial_index_relative_to_this_subject
                                data_of_a_modal.append([trial_directory, absolute_indices, relative_indices, session_name])

                            if len(session_indices) > 1:
                                initial_index_relative_to_this_subject += length_list[i]

                    data_dict[partition].extend(data_of_a_modal)

            if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                eeg_raw_mean = trial_wise_eeg_mean_cache / num_samples

                for eeg_raw_path in eeg_raw_path_list:
                    trial_wise_eeg_std_cache += np.sum(np.square(np.load(eeg_raw_path) - eeg_raw_mean), axis=0)

                eeg_raw_std = np.sqrt(trial_wise_eeg_std_cache / num_samples)
                normalize_dict[partition]['mean'] = eeg_raw_mean
                normalize_dict[partition]['std'] = eeg_raw_std

        return data_dict, normalize_dict

    def make_length_dict(self, subject_id_of_all_folds, partition_dictionary):

        # Get the partition-wise subject dictionary.
        subject_id_of_all_partitions = self.partition_train_validate_test_for_subjects(
            subject_id_of_all_folds, partition_dictionary)

        # Initialize the dictionary to be outputted.
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
        # "partition_dictionary" is set by the user, while "subject_id_of_all_folds" is generated according to the
        # evaluation of subject-wise trial number. They should be equal.
        # For example, partition_dictionary = {train: 6, val: 3, test:1} has ten folds. So that the list subject_id_of_all_folds
        #   should also contain 10 elements.
        assert len(subject_id_of_all_folds) == np.sum([value for value in partition_dictionary.values()])

        # Assign the subject id according to the dictionary.
        subject_id_of_all_partitions = {
            'train': subject_id_of_all_folds[0:partition_dictionary['train']],
            'validate': subject_id_of_all_folds[partition_dictionary['train']:
                                                partition_dictionary['train'] + partition_dictionary['validate']],
            'test': subject_id_of_all_folds[-partition_dictionary['test']:]
        }

        return subject_id_of_all_partitions


class NFoldMahnobArrangerLOSO(VideoEmoRegressionArranger):
    r"""
    A class to prepare files according to the Leave-one-subject-out manner.
    """

    def __init__(self, dataset_load_path, dataset_folder, modality, normalize_eeg_raw=True, num_electrodes=32, window_sec=24, hop_size_sec=8, continuous_label_frequency=4, include_session_having_no_continuous_label=True, feature_extraction=False):

        # The root directory of the dataset.
        super().__init__(dataset_load_path=dataset_load_path, dataset_folder=dataset_folder)

        # Load the dataset information
        self.dataset_info = self.get_dataset_info()

        self.num_electrodes = num_electrodes
        self.normalize_eeg_raw = normalize_eeg_raw

        # Regression or Classification?
        self.include_session_having_no_continuous_label = include_session_having_no_continuous_label

        self.depth = window_sec * continuous_label_frequency
        self.step_size = hop_size_sec * continuous_label_frequency

        # Frame, EEG, which one or both?
        self.modality = modality
        self.get_modality_list()

        # Get the sessions having continuous labels.
        self.sessions_having_continuous_label = self.get_session_indices_having_continuous_label()
        self.feature_extraction = feature_extraction

    def get_modality_list(self):

        # if self.job == "reg_v":
        self.modality.append("continuous_label")

    def get_session_indices_having_continuous_label(self):
        r"""
        Get the session indices having continuous labels.
        :return: (list), the indices indicating which sessions have continuous labels.
        """
        if self.include_session_having_no_continuous_label:
            indices = np.where(np.asarray(self.dataset_info['having_eeg']) == 1)[0]
        else:
            indices = np.where(self.dataset_info['having_continuous_label'] == 1)[0]
        return indices

    def make_data_dict(self, trial_id_of_all_partitions):

        # Initialize the dictionary to be outputed.
        data_dict = {key: [] for key in trial_id_of_all_partitions}
        length_dict = {key: [] for key in trial_id_of_all_partitions}
        normalize_dict = {key: {} for key in trial_id_of_all_partitions}

        for partition, trial_id_of_a_partition in trial_id_of_all_partitions.items():

            data_of_a_partition = []
            length_of_a_partition = []
            if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                trial_wise_eeg_mean_cache = np.zeros((self.num_electrodes,))
                trial_wise_eeg_std_cache = np.zeros((self.num_electrodes,))
                num_samples = 0
                eeg_raw_path_list = []

            trial_indices = trial_id_of_a_partition

            if len(trial_indices) == 1:
                length_list = [self.dataset_info['refined_processed_length'][trial_indices[0]]]
                trial_name_list = [self.dataset_info['session_name'][trial_indices[0]]]
                directory_list = [os.path.join(self.root_directory, self.npy_folder, session_name) for
                                  session_name in trial_name_list]
            else:
                length_list = list(itemgetter(*trial_indices)(self.dataset_info['refined_processed_length']))
                trial_name_list = list(itemgetter(*trial_indices)(self.dataset_info['session_name']))
                directory_list = [os.path.join(self.root_directory, self.npy_folder, session_name) for
                                  session_name in trial_name_list]

            for i in range(len(trial_indices)):
                trial_name = trial_name_list[i]
                length = length_list[i]
                trial_directory = directory_list[i]

                if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                    eeg_raw_path = os.path.join(trial_directory, "eeg_raw.npy")
                    intermediate = np.load(eeg_raw_path)
                    num_samples += intermediate.shape[0]
                    trial_wise_eeg_mean_cache += np.sum(intermediate, axis=0)
                    eeg_raw_path_list.append(eeg_raw_path)

                if self.feature_extraction:
                    depth = length
                else:
                    depth = self.depth

                start = 0
                end = start + depth

                while end <= length:
                    relative_indices = np.arange(start, end)
                    data_of_a_partition.append([trial_directory, relative_indices, length, trial_name])
                    start += self.step_size
                    end = start + depth

                if (length - depth) % self.step_size != 0:
                    start = length - depth
                    end = length
                    relative_indices = np.arange(start, end)
                    data_of_a_partition.append([trial_directory, relative_indices, length, trial_name])

            data_dict[partition].extend(data_of_a_partition)

            if "eeg_raw" in self.modality and self.normalize_eeg_raw:
                eeg_raw_mean = trial_wise_eeg_mean_cache / num_samples

                for eeg_raw_path in eeg_raw_path_list:
                    trial_wise_eeg_std_cache += np.sum(np.square(np.load(eeg_raw_path) - eeg_raw_mean), axis=0)

                eeg_raw_std = np.sqrt(trial_wise_eeg_std_cache / num_samples)
                normalize_dict[partition]['mean'] = eeg_raw_mean
                normalize_dict[partition]['std'] = eeg_raw_std

        return data_dict, normalize_dict

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

        # For Leave-one-subject-out scenario:
        # Also output the session id of all folds for convenience.
        trial_id_of_all_folds = {subject_id_of_one_fold[0]: np.hstack([self.sessions_having_continuous_label[np.where(
            self.dataset_info['subject_id'][self.sessions_having_continuous_label] == subject_id)[0]]
                                              for subject_id in subject_id_of_one_fold])
                                   for subject_id_of_one_fold in subject_id_of_all_folds}

        return subject_id_of_all_folds, trial_id_of_all_folds

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


class MAHNOBDatasetCls(Dataset):
    def __init__(self, config, data_list, normalize_dict=None, modality=['frame'], emotion_dimension=['Valence'],
                 continuous_label_frequency=4,
                 eegnet_window_sec=2, eegnet_stride_sec=0.25, frame_size=48, crop_size=40, normalize_eeg_raw=0,
                 window_sec=24, hop_size=8, time_delay=0, class_labels=None, mode='train', feature_extraction=False):

        self.frame_size = frame_size
        self.crop_size = crop_size
        self.mode = mode
        self.data_list = data_list
        self.normalize_dict = normalize_dict
        self.normalize_eeg_raw = normalize_eeg_raw
        self.ratio = config['downsampling_interval_dict']
        self.modality = modality
        self.feature_extraction = feature_extraction
        self.emotion_dimension = emotion_dimension
        self.class_labels = class_labels,
        self.get_3D_transforms()
        self.targets = self.get_targets()

        self.frame_window_length = int(window_sec * config['frequency_dict']['frame'])
        self.frame_stride = int(hop_size * config['frequency_dict']['frame'])

        self.eeg_window_length = int(window_sec * config['frequency_dict']['eeg_raw'])
        self.eeg_stride = int(hop_size * config['frequency_dict']['eeg_raw'])

    def get_targets(self):

        targets = np.empty(0, dtype=np.int64)
        for item in self.data_list:
            targets = np.append(targets, int(self.class_labels[0][item[-1]]["Valence"]))

        return targets

    def get_3D_transforms(self):
        normalize = transforms3D.GroupNormalize([0.5077, 0.5077, 0.5077], [0.2544, 0.2544, 0.2544])
        normalize_eeg_image = transforms3D.GroupNormalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        if self.mode == 'train':
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    transforms3D.GroupNumpyToPILImage(0),
                    transforms3D.GroupRandomCrop(self.frame_size, self.crop_size),
                    transforms3D.GroupRandomHorizontalFlip(),
                    transforms3D.Stack(),
                    transforms3D.ToTorchFormatTensor(),
                    normalize
                ])

            if "eeg_image" in self.modality:
                self.eeg_transforms = transforms.Compose([
                    transforms3D.GroupWhiteNoiseByPCA(std_multiplier=0.1, num_components=2),
                    transforms3D.ToTorchFormatTensor(),
                    normalize_eeg_image
                ])

            if "eeg_raw" in self.modality:
                eeg_raw_transforms = []
                if self.normalize_eeg_raw:
                    eeg_raw_transforms.append(transforms3D.GroupEegRawDataNormalize(
                        mean=self.normalize_dict[self.mode]['mean'], std=self.normalize_dict[self.mode]['std']))
                eeg_raw_transforms.append(transforms3D.GroupEegRawToTensor())

                self.eeg_raw_transforms = transforms.Compose(eeg_raw_transforms)

        else:
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    transforms3D.GroupNumpyToPILImage(0),
                    transforms3D.GroupCenterCrop(self.crop_size),
                    transforms3D.Stack(),
                    transforms3D.ToTorchFormatTensor(),
                    normalize
                ])

            if "eeg_image" in self.modality:
                self.eeg_transforms = transforms.Compose([
                    transforms3D.ToTorchFormatTensor(),
                    normalize_eeg_image
                ])

            if "eeg_raw" in self.modality:
                self.eeg_raw_transforms = transforms.Compose([
                    transforms3D.GroupEegRawToTensor()
                ])

    def get_frame_indices(self, indices):
        x = 0
        if self.mode == 'train':
            x = random.randint(0, self.ratio['frame'] - 1)
        indices = indices * self.ratio['frame'] + x
        return indices

    def get_eeg_indices(self, indices):
        start = indices[0] * self.ratio['eeg_raw']
        end = (indices[-1] + 1) * self.ratio['eeg_raw']
        indices = np.arange(start, end)
        return indices

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def load_data(directory, indices, filename):
        filename = os.path.join(directory, filename)
        frames = np.load(filename, mmap_mode='c')[indices]
        return frames

    def resample_eeg_raw(self, eeg_raw):
        eeg_raw_matrix = eeg_raw[np.newaxis, :]
        return eeg_raw_matrix

    def __getitem__(self, index):
        directory = self.data_list[index][0]
        absolute_indices = self.data_list[index][1]
        relative_indices = self.data_list[index][2]

        session = self.data_list[index][3]
        new_indices = self.get_frame_indices(relative_indices)

        eeg_indices = self.get_eeg_indices(relative_indices)

        features = {}

        if "frame" in self.modality:
            frames = self.load_data(directory, new_indices, "frame.npy")
            frames = self.image_transforms(frames)
            features.update({'frame': frames})

        if "eeg_image" in self.modality:
            eeg_images = self.load_data(directory, relative_indices, "eeg_image.npy")
            eeg_images = self.eeg_transforms(eeg_images)
            features.update({'eeg_image': eeg_images})

        if "eeg_raw" in self.modality:
            eeg_raws = self.load_data(directory, eeg_indices, "eeg_raw.npy").transpose()
            eeg_raws = self.resample_eeg_raw(eeg_raws)
            eeg_raws = np.asarray(eeg_raws, dtype=np.float32)
            eeg_raws = self.eeg_raw_transforms(eeg_raws)
            features.update({'eeg_raw': eeg_raws})

        if "eeg_psd" in self.modality:
            eeg_psds = self.load_data(directory, relative_indices, "eeg_psd.npy").transpose()
            eeg_psds = np.asarray(eeg_psds, dtype=np.float32)
            features.update({'eeg_raw': eeg_psds})

        labels = self.class_labels[0][session]["Valence"]

        return features, labels, absolute_indices, session


class MAHNOBDataset(Dataset):
    def __init__(self, config, data_list, normalize_dict=None, modality=['frame'], emotion_dimension=['Valence'], continuous_label_frequency=4,
                 eegnet_window_sec=2, eegnet_stride_sec=0.25, frame_size=48, crop_size=40, normalize_eeg_raw=0,
                 window_sec=24, hop_size=8, time_delay=0, class_labels=None, mode='train', feature_extraction=False,
                 load_knowledge=False, knowledge_path='', fold=0):
        self.frame_size = frame_size
        self.crop_size = crop_size
        self.mode = mode
        self.data_list = data_list
        self.normalize_dict = normalize_dict
        self.normalize_eeg_raw = normalize_eeg_raw
        self.ratio = config['downsampling_interval_dict']
        self.modality = modality
        self.feature_extraction = feature_extraction
        self.emotion_dimension = emotion_dimension
        self.time_delay = np.int(time_delay * 4)
        self.get_3D_transforms()
        self.class_label = class_labels
        self.continuous_label_frequency = continuous_label_frequency
        self.eegnet_window_sec = eegnet_window_sec
        self.eeg_window_length = int(eegnet_window_sec * config['frequency_dict']['eeg_raw'])
        self.stride = int(eegnet_stride_sec * config['frequency_dict']['eeg_raw'])
        self.num_eegnet_windows = window_sec * config['frequency_dict']['continuous_label']

        self.load_knowledge = load_knowledge
        self.knowledge_path = os.path.join(knowledge_path, str(fold))

    def get_3D_transforms(self):
        normalize = transforms3D.GroupNormalize([0.5077, 0.5077, 0.5077], [0.2544, 0.2544, 0.2544])
        normalize_eeg_image = transforms3D.GroupNormalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        if self.mode == 'train':
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    transforms3D.GroupNumpyToPILImage(0),
                    transforms3D.GroupRandomCrop(self.frame_size, self.crop_size),
                    transforms3D.GroupRandomHorizontalFlip(),
                    transforms3D.Stack(),
                    transforms3D.ToTorchFormatTensor(),
                    normalize
                ])

            if "eeg_image" in self.modality:
                self.eeg_transforms = transforms.Compose([
                    transforms3D.GroupWhiteNoiseByPCA(std_multiplier=0.1, num_components=2),
                    transforms3D.ToTorchFormatTensor(),
                    normalize_eeg_image
                ])

            if "eeg_raw" in self.modality:
                eeg_raw_transforms = []
                if self.normalize_eeg_raw:
                    eeg_raw_transforms.append(transforms3D.GroupEegRawDataNormalize(
                        mean=self.normalize_dict[self.mode]['mean'], std=self.normalize_dict[self.mode]['std']))
                eeg_raw_transforms.append(transforms3D.GroupEegRawToTensor())

                self.eeg_raw_transforms = transforms.Compose(eeg_raw_transforms)

        else:
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    transforms3D.GroupNumpyToPILImage(0),
                    transforms3D.GroupCenterCrop(self.crop_size),
                    transforms3D.Stack(),
                    transforms3D.ToTorchFormatTensor(),
                    normalize
                ])

            if "eeg_image" in self.modality:
                self.eeg_transforms = transforms.Compose([
                    transforms3D.ToTorchFormatTensor(),
                    normalize_eeg_image
                ])

            if "eeg_raw" in self.modality:
                self.eeg_raw_transforms = transforms.Compose([
                    transforms3D.GroupEegRawToTensor()
                ])

    def get_frame_indices(self, indices):
        x = 0
        origin_indices = indices
        if not self.feature_extraction:
            if self.mode == 'train':
                x = random.randint(0, self.ratio['frame'] - 1)
            indices = indices * self.ratio['frame'] + x
            return indices
        else:
            # In order to extract the frame-wise features, the indices cannot be downsampled.
            non_downsampled_indices = np.empty(0,)
            for x in range(self.ratio['frame']):
                indices = origin_indices * self.ratio['frame'] + x
                non_downsampled_indices = np.append(non_downsampled_indices, indices)
            non_downsampled_indices = np.asarray(np.sort(non_downsampled_indices), dtype=np.int64)
            return non_downsampled_indices

    def get_eeg_indices(self, indices):
        start = indices[0] * self.ratio['eeg_raw']
        end = (indices[-1] + 1 + self.continuous_label_frequency * self.eegnet_window_sec) * self.ratio['eeg_raw']
        indices = np.arange(start, end)
        return indices

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def load_data(directory, indices, filename):
        filename = os.path.join(directory, filename)
        frames = np.load(filename, mmap_mode='c')[indices]
        return frames

    def resample_eeg_raw(self, eeg_raw):
        eeg_raw_matrix = np.zeros((self.num_eegnet_windows, 1, 32, self.eeg_window_length))
        start = 0
        end = self.eeg_window_length
        for i in range(self.num_eegnet_windows):
            eeg_raw_matrix[i] = eeg_raw[:, start:end]
            start += self.stride
            end = start + self.eeg_window_length
        return eeg_raw_matrix

    def __getitem__(self, index):
        directory = self.data_list[index][0]
        absolute_indices = self.data_list[index][1]
        relative_indices = self.data_list[index][2]

        session = self.data_list[index][3]
        new_indices = self.get_frame_indices(relative_indices)

        eeg_indices = self.get_eeg_indices(relative_indices)

        features = {}

        if "frame" in self.modality:
            frames = self.load_data(directory, new_indices, "frame.npy")
            frames = self.image_transforms(frames)
            features.update({'frame': frames})

        if "eeg_image" in self.modality:
            eeg_images = self.load_data(directory, relative_indices, "eeg_image.npy")
            eeg_images = self.eeg_transforms(eeg_images)
            features.update({'eeg_image': eeg_images})

        if "eeg_raw" in self.modality:
            eeg_raws = self.load_data(directory, eeg_indices, "eeg_raw.npy").transpose()
            eeg_raws = self.resample_eeg_raw(eeg_raws)
            eeg_raws = np.asarray(eeg_raws, dtype=np.float32)
            eeg_raws = self.eeg_raw_transforms(eeg_raws)
            features.update({'eeg_raw': eeg_raws})

        if "eeg_psd" in self.modality:
            eeg_psds = self.load_data(directory, relative_indices, "eeg_psd.npy").transpose()
            eeg_psds = np.asarray(eeg_psds, dtype=np.float32)
            features.update({'eeg_psd': eeg_psds})

        if self.load_knowledge:
            filename = os.path.join(session + ".npy")
            knowledges = self.load_data(self.knowledge_path, relative_indices, filename)
            features.update({'knowledge': knowledges})

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


class MAHNOBDatasetTrial(Dataset):
    def __init__(self, config, data_list, normalize_dict=None, modality=['frame'], emotion_dimension=['Valence'], continuous_label_frequency=4,
                 eegnet_window_sec=2, eegnet_stride_sec=0.25, frame_size=48, crop_size=40, normalize_eeg_raw=0,
                 window_sec=24, hop_size=8, time_delay=0, class_labels=None, mode='train', feature_extraction=False,
                 load_knowledge=False, knowledge_path='', fold=0):
        self.frame_size = frame_size
        self.crop_size = crop_size
        self.mode = mode
        self.data_list = data_list
        self.normalize_dict = normalize_dict
        self.normalize_eeg_raw = normalize_eeg_raw
        self.ratio = config['downsampling_interval_dict']
        self.modality = modality
        self.feature_extraction = feature_extraction
        self.emotion_dimension = emotion_dimension
        self.time_delay = np.int(time_delay * 4)
        self.get_3D_transforms()
        self.class_label = class_labels
        self.continuous_label_frequency = continuous_label_frequency
        self.eegnet_window_sec = eegnet_window_sec
        self.eeg_window_length = int(eegnet_window_sec * config['frequency_dict']['eeg_raw'])
        self.stride = int(eegnet_stride_sec * config['frequency_dict']['eeg_raw'])
        self.num_eegnet_windows = window_sec * config['frequency_dict']['continuous_label']

        self.load_knowledge = load_knowledge
        self.knowledge_path = os.path.join(knowledge_path, str(fold))

    def get_3D_transforms(self):
        normalize = transforms3D.GroupNormalize([0.5077, 0.5077, 0.5077], [0.2544, 0.2544, 0.2544])
        normalize_eeg_image = transforms3D.GroupNormalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        if self.mode == 'train':
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    transforms3D.GroupNumpyToPILImage(0),
                    transforms3D.GroupRandomCrop(self.frame_size, self.crop_size),
                    transforms3D.GroupRandomHorizontalFlip(),
                    transforms3D.Stack(),
                    transforms3D.ToTorchFormatTensor(),
                    normalize
                ])

            if "eeg_image" in self.modality:
                self.eeg_transforms = transforms.Compose([
                    transforms3D.GroupWhiteNoiseByPCA(std_multiplier=0.1, num_components=2),
                    transforms3D.ToTorchFormatTensor(),
                    normalize_eeg_image
                ])

            if "eeg_raw" in self.modality:
                eeg_raw_transforms = []
                if self.normalize_eeg_raw:
                    eeg_raw_transforms.append(transforms3D.GroupEegRawDataNormalize(
                        mean=self.normalize_dict[self.mode]['mean'], std=self.normalize_dict[self.mode]['std']))
                eeg_raw_transforms.append(transforms3D.GroupEegRawToTensor())

                self.eeg_raw_transforms = transforms.Compose(eeg_raw_transforms)

        else:
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    transforms3D.GroupNumpyToPILImage(0),
                    transforms3D.GroupCenterCrop(self.crop_size),
                    transforms3D.Stack(),
                    transforms3D.ToTorchFormatTensor(),
                    normalize
                ])

            if "eeg_image" in self.modality:
                self.eeg_transforms = transforms.Compose([
                    transforms3D.ToTorchFormatTensor(),
                    normalize_eeg_image
                ])

            if "eeg_raw" in self.modality:
                self.eeg_raw_transforms = transforms.Compose([
                    transforms3D.GroupEegRawToTensor()
                ])

    def get_frame_indices(self, indices):
        x = 0
        origin_indices = indices
        if not self.feature_extraction:
            if self.mode == 'train':
                x = random.randint(0, self.ratio['frame'] - 1)
            indices = indices * self.ratio['frame'] + x
            return indices
        else:
            non_downsampled_indices = np.empty(0,)
            for x in range(self.ratio['frame']):
                indices = origin_indices * self.ratio['frame'] + x
                non_downsampled_indices = np.append(non_downsampled_indices, indices)
            non_downsampled_indices = np.asarray(np.sort(non_downsampled_indices), dtype=np.int64)
            return non_downsampled_indices

    def get_eeg_indices(self, indices):
        start = indices[0] * self.ratio['eeg_raw']
        end = (indices[-1] + 1 + self.continuous_label_frequency * self.eegnet_window_sec) * self.ratio['eeg_raw']
        indices = np.arange(start, end)
        return indices

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def load_data(directory, indices, filename):
        filename = os.path.join(directory, filename)
        frames = np.load(filename, mmap_mode='c')[indices]
        return frames

    def resample_eeg_raw(self, eeg_raw):
        eeg_raw_matrix = np.zeros((self.num_eegnet_windows, 1, 32, self.eeg_window_length))
        start = 0
        end = self.eeg_window_length
        for i in range(self.num_eegnet_windows):
            eeg_raw_matrix[i] = eeg_raw[:, start:end]
            start += self.stride
            end = start + self.eeg_window_length
        return eeg_raw_matrix

    def __getitem__(self, index):
        directory = self.data_list[index][0]
        relative_indices = self.data_list[index][1]
        length = self.data_list[index][2]
        trial = self.data_list[index][3]

        new_indices = self.get_frame_indices(relative_indices)

        eeg_indices = self.get_eeg_indices(relative_indices)

        features = {}

        if "frame" in self.modality:
            frames = self.load_data(directory, new_indices, "frame.npy")
            frames = self.image_transforms(frames)
            features.update({'frame': frames})

        if "eeg_image" in self.modality:
            eeg_images = self.load_data(directory, relative_indices, "eeg_image.npy")
            eeg_images = self.eeg_transforms(eeg_images)
            features.update({'eeg_image': eeg_images})

        if "eeg_raw" in self.modality:
            eeg_raws = self.load_data(directory, eeg_indices, "eeg_raw.npy").transpose()
            eeg_raws = self.resample_eeg_raw(eeg_raws)
            eeg_raws = np.asarray(eeg_raws, dtype=np.float32)
            eeg_raws = self.eeg_raw_transforms(eeg_raws)
            features.update({'eeg_raw': eeg_raws})

        if "eeg_psd" in self.modality:
            eeg_psds = self.load_data(directory, relative_indices, "eeg_psd.npy").transpose()
            eeg_psds = np.asarray(eeg_psds, dtype=np.float32)
            features.update({'eeg_psd': eeg_psds})

        if self.load_knowledge and self.mode != "test":
            filename = os.path.join(trial + ".npy")
            knowledges = self.load_data(self.knowledge_path, relative_indices, filename)
            features.update({'knowledge': knowledges})

        if self.class_label is None:
            # Regression on Valence
            labels = self.load_data(directory, relative_indices, "continuous_label.npy")

            # time delay
            labels = np.concatenate(
                (labels[self.time_delay:, :],
                 np.repeat(labels[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)
        else:
            labels = self.class_label[trial]["Valence"]

        return features, labels, relative_indices, trial, length
