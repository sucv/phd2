from base.preprocessing import GenericVideoPreprocessing
from base.utils import get_filename_from_a_folder_given_extension, get_filename_from_full_path, get_video_length, save_pkl_file, copy_file
from base.video import OpenFaceController
from base.utils import dict_combine
from base.metric import concordance_correlation_coefficient_centering
from base.video import combine_annotated_clips, change_video_fps
from project.emotion_regression_on_semaine.utils import read_semaine_xml_for_dataset_info, continuous_label_to_csv


import os
import json
from tqdm import tqdm
import pickle
import re
import xml.etree.ElementTree as et

import numpy as np
import scipy.io as sio
import cv2
import mne
import pandas as pd
from PIL import Image


class PreprocessingSEMAINE(GenericVideoPreprocessing):
    def __init__(self, opts):
        super().__init__(opts)

        self.filename_pattern = config['filename_pattern']

        # To how many frames is a label corresponds.
        self.label_to_video_ratio = config['downsampling_interval_dict']['frame']

        # The intermediate folder to save the read continuous labels.
        # This is for time saving because there are too many labels!
        # Cannot read them everytime when debugging.
        self.intermediate_folder = config['intermediate_folder']

        # The config for the powerful openface.
        self.openface_config = config['openface_config']

        self.fold_number = config['fold_number_preprocessing']

        # The dictionary to store the session_id, subject_id, trial_id, data length, trimming range, etc.
        self.dataset_info = self.get_subject_trial_info()

        # The total of the sessions of the dataset.
        self.session_number = self.count_session()

        self.continuous_label_list = self.get_continuous_label()

        # Obtain the data length for each trial.
        self.dataset_info['continuous_label_length'] = self.get_continuous_label_length()

        # Obtain the trimmed length of the video for each trial.
        self.dataset_info['trim_length'] = self.get_video_trimmed_length()

        # Obtain the trimming range of the video for each trial.
        self.dataset_info['trim_range'] = self.get_video_trimming_range()

        self.target_fps = config['target_fps']


        # # Carry out the video preprocessing,
        # # Alter the fps ---> Trim the video ---> Extract the facial fiducial points
        # #  ---> crop the video frames ---> save to images
        # self.video_preprocessing()

        # Carryout the label preprocessing.
        # Perform the CCC centering ---> Save both the continuous label
        # and the success indicator to csv files.
        self.label_preprocessing()

        # Save the dataset information. It is important because in can provide a consistent processing order.
        # For example, in Windows and Ubuntu, the sort can has different results, which results in a different
        # subject_id, trial_id orders. If the openface and sort are performed in Windows, and the sort is
        # performed in Ubuntu again, then the dataset information from the two system will be different, so that
        # the information will not correspond to the openface output.
        self.save_dataset_info()

    def count_session(self):
        r"""
        Count the total of the sessions.
        :return: (int), the total of the sessions.
        """
        session_number = len(self.dataset_info["session_id"])
        return session_number

    def get_subject_trial_info(self):
        r"""
        Get the session_id, subject_id, subject_role and the feeltrace indicators.
        :return: (dict), the dictionary of the dataset information.
        """
        dataset_info = {"session_id": [],
                        "subject_id": [],
                        "subject_role": [],
                        "feeltrace_bool": []}

        directory = os.path.join(self.root_directory, "Sessions")

        for sub_folder in os.listdir(directory):
            xml_file = et.parse(os.path.join(directory, sub_folder, self.filename_pattern['session_log'])).getroot()

            user_info = read_semaine_xml_for_dataset_info(xml_file, "User")
            dataset_info = dict_combine(dataset_info, user_info)

            operator_info = read_semaine_xml_for_dataset_info(xml_file, "Operator")
            dataset_info = dict_combine(dataset_info, operator_info)

        dataset_info = {key: np.asarray(value) for key, value in dataset_info.items()}

        # Note, the sort_indices can be different between Windows and Ubuntu!!
        # So that the dataset_info has to be saved and copied to a different OS for consistency.
        sort_indices = np.argsort(dataset_info['subject_id'])
        dataset_info = {key: value[sort_indices] for key, value in dataset_info.items()}
        dataset_info = self.generate_trial_info(dataset_info)

        # If the dataset_info already exists, read it! This is for the above mentioned consistency!
        dataset_info_filename = os.path.join(self.root_directory, "dataset_info.pkl")
        if os.path.isfile(dataset_info_filename):
            with open(dataset_info_filename, 'rb') as f:
                existing_dataset_info = pickle.load(f)
            dataset_info.update(existing_dataset_info)

        return dataset_info

    @staticmethod
    def generate_trial_info(dataset_info):
        r"""
        Generate unrepeated trial index given the subject index.
            The Semaine has not a existing records on trials of a same subject.
            Therefore, this function is used to generate so that the Subject-trial index is unique for each session.
        :param dataset_info: (dict), the dictionary recording the information of the dataset.
        :return: (dict), a new dictionary having a new key named "trial_id".
        """
        trial_info = np.zeros_like(dataset_info['subject_id'])
        unique_subject_array, count = np.unique(dataset_info['subject_id'], return_counts=True)

        for idx, subject in enumerate(unique_subject_array):
            indices = np.where(dataset_info['subject_id'] == subject)[0]
            trial_info[indices] = np.arange(1, count[idx] + 1, 1)

        dataset_info['trial_id'] = trial_info
        return dataset_info

    def get_video_trimmed_length(self):
        r"""
        :return: the frame range of a video that corresponds to its annotation (dict).
        By default, the range starts from 0.
        """
        lengths = self.dataset_info['continuous_label_length'] * self.label_to_video_ratio
        return lengths

    def get_video_trimming_range(self):
        zero = np.zeros((len(self.dataset_info['trim_length']), 1), dtype=int)

        ranges = np.c_[zero, self.dataset_info['trim_length']]
        ranges = ranges[:, np.newaxis, :]

        return ranges

    def get_label_dict_by_pattern(self, pattern):
        r"""
        Get the dictionary storing the continuous labels.
        :param pattern: (string), the pattern of the files.
        :return: (dict), the filename dictionary of the continuous labels for users and operators.
        """
        folder = os.path.join(self.root_directory, "Sessions")
        dataset_info = self.dataset_info
        label_dict = {key: [] for key in self.emotion_dimension}
        role_dict = {0: "User", 1: "Operator"}

        for i in range(self.session_number):
            if dataset_info['feeltrace_bool'][i]:
                directory = os.path.join(folder, str(dataset_info['session_id'][i]))

                for emotion in self.emotion_dimension:

                    file_pattern = pattern[role_dict[dataset_info['subject_role'][i]]][emotion]
                    reg_compile = re.compile(file_pattern)

                    filename = sorted([os.path.join(directory, file) for file
                                       in os.listdir(directory) if reg_compile.match(file)])
                    if filename:
                        label_dict[emotion].append(filename)

        return label_dict

    def get_video_list_by_pattern(self, pattern):
        r"""
        Get the dictionary storing the videos having the continuous labels.
        :param pattern: (string), the pattern of the files.
        :return: (list), the filename list of the videos. Both the users and operators are taken as subjects,
            therefore a list not dictionary is suitable.
        """
        folder = os.path.join(self.root_directory, "Sessions")
        dataset_info = self.dataset_info
        video_list = []
        role_dict = {0: "User", 1: "Operator"}

        for index in range(self.session_number):
            if dataset_info['feeltrace_bool'][index]:
                directory = os.path.join(folder, str(dataset_info['session_id'][index]))
                file_pattern = pattern[role_dict[dataset_info['subject_role'][index]]]
                reg_compile = re.compile(file_pattern)
                filename = [os.path.join(directory, file) for file
                            in os.listdir(directory) if reg_compile.match(file)][0]
                video_list.append(filename)

        return video_list

    def get_intermediate_continuous_label(self):
        r"""
        Save the continuous label to disk from the memory. For time-saving during the dubbing.
        :return: (string), (string), the filename of the files saving the intermediate continuous labels and their length.
        """
        intermediate_label_filename = os.path.join(self.root_directory, self.intermediate_folder,
                                                   "intermediate_label.pkl")
        intermediate_label_length_filename = os.path.join(self.root_directory, self.intermediate_folder,
                                                          "intermediate_label_length.pkl")
        os.makedirs(os.path.join(self.root_directory, self.intermediate_folder), exist_ok=True)
        label_dict = {key: [] for key in self.emotion_dimension}
        label_length_dict = {key: [] for key in self.emotion_dimension}
        label_file = self.get_label_dict_by_pattern(self.filename_pattern['continuous_label'])

        if not os.path.isfile(intermediate_label_filename):
            for emotion in self.emotion_dimension:
                emotion_label_file = label_file[emotion]
                subject_level_list, subject_level_length_list = [], []

                for count, subject_level in enumerate(emotion_label_file):
                    rater_level_list = []
                    rater_level_list_length = []
                    for rater_level in subject_level:
                        label = np.loadtxt(rater_level)[:, 1]
                        rater_level_list_length.append(len(label))
                        rater_level_list.append(label)
                    min_length = min(rater_level_list_length)
                    subject_level_length_list.append(min_length)
                    rater_level_list = [label[:min_length] for label in rater_level_list]
                    subject_level_list.append(np.stack(rater_level_list))
                    print(count)

                label_dict[emotion] = subject_level_list
                label_length_dict[emotion] = subject_level_length_list

            with open(intermediate_label_filename, 'wb') as f:
                pickle.dump(label_dict, f)

            with open(intermediate_label_length_filename, 'wb') as f:
                pickle.dump(label_length_dict, f)

        return intermediate_label_filename, intermediate_label_length_filename

    def get_continuous_label_length(self):
        r"""
        Read the length of the continuous label.
        :return: (ndarray), the minimum length across the two emotional dimensions for each session.
        """
        _, intermediate_label_length_filename = self.get_intermediate_continuous_label()

        with open(intermediate_label_length_filename, 'rb') as f:
            intermediate_label_length = pickle.load(f)
        min_length = np.min(np.vstack([value for value in intermediate_label_length.values()]), axis=0)

        return min_length

    def get_continuous_label(self):
        r"""
        Read the continuous labels.
        :return: (dict), the dict saving the continuous labels. It is really fast when directly reads them from a pkl file.
        """
        intermediate_label_filename = os.path.join(self.root_directory, self.intermediate_folder,
                                                   "intermediate_label.pkl")
        label_dict = {key: [] for key in self.emotion_dimension}

        min_length = self.get_continuous_label_length()

        with open(intermediate_label_filename, 'rb') as f:
            intermediate_label = pickle.load(f)

        for emotion in self.emotion_dimension:
            mat_cell = np.squeeze(intermediate_label[emotion])

            # This block is for reading a pkl file saving a dictionary. A directly reading without forloop
            # can only obtain the first list.
            label_list = []
            for index in range(len(min_length)):
                label_list.append(mat_cell[index][:, :min_length[index]])

            label_dict[emotion] = label_list

        return label_dict

    def label_ccc_centering(self):
        r"""
                Carry out the label preprocessing. Here, since multiple raters are available, therefore
                    concordance_correlation_coefficient_centering has to be performed.
                """
        centered_continuous_label_dict = {key: [] for key in self.emotion_dimension}

        for emotion in self.emotion_dimension:
            continuous_labels = self.continuous_label_list[emotion]
            for continuous_label in continuous_labels:
                centered_continuous_label = concordance_correlation_coefficient_centering(continuous_label)
                centered_continuous_label_dict[emotion].append(np.float32(np.mean(centered_continuous_label, axis=0)))

        return centered_continuous_label_dict

    def video_preprocessing(self):
        r"""
        Carry out the video preprocessing.
        """
        video_list = self.get_video_list_by_pattern(self.filename_pattern['video'])

        # Change the fps of the video to a integer.
        video_list = change_video_fps(video_list, self.target_fps)

        # Pick only the annotated clips from a complete video.
        video_list = combine_annotated_clips(
            video_list, self.dataset_info['trim_range'], direct_copy=False, visualize=False)

        output_directory = os.path.join(self.root_directory, self.openface_output_folder)
        os.makedirs(output_directory, exist_ok=True)

        # Extract facial landmark, warp, crop, and save each frame.
        openface = OpenFaceController(openface_path=self.openface_config['openface_directory'],
                                      output_directory=output_directory)
        video_list = openface.process_video_list(video_list=video_list, dataset_info=self.dataset_info, **self.openface_config)

        # Save the static folders
        self.dataset_info['processed_folder'] = video_list

    def label_preprocessing(self):

        centered_continuous_label_dict = self.label_ccc_centering()


        continuous_label_to_csv(
            self.root_directory, self.output_folder, centered_continuous_label_dict,
            self.dataset_info)

    def create_npy_for_continuous_label(self):

        pointer = 0
        for i, folder in tqdm(enumerate(range(self.session_number))):
            if self.dataset_info['having_continuous_label'][i]:
                npy_directory = self.dataset_info['npy_output_folder'][i]
                os.makedirs(npy_directory, exist_ok=True)
                npy_filename_frame = os.path.join(npy_directory, "continuous_label.npy")
                if not os.path.isfile(npy_filename_frame):
                    continuous_label_length = self.dataset_info['refined_processed_length'][i]
                    continuous_label = self.continuous_label_list[pointer][:continuous_label_length]

                    with open(npy_filename_frame, 'wb') as f:
                        np.save(f, continuous_label)

                    pointer += 1
                else:
                    pointer += 1




if __name__ == "__main__":
    with open("config_semaine") as config_file:
        config = json.load(config_file)

    pre = PreprocessingSEMAINE(config)