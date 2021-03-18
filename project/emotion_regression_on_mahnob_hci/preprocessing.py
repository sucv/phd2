from base.preprocessing import GenericVideoPreprocessing
from base.utils import get_filename_from_a_folder_given_extension, get_filename_from_full_path, get_video_length, save_pkl_file, copy_file
from base.video import OpenFaceController
from base.utils import load_single_pkl
from base.video import combine_annotated_clips, change_video_fps
from project.emotion_regression_on_mahnob_hci.utils import read_start_end_from_mahnob_tsv
from project.emotion_regression_on_mahnob_hci.eeg import EegMahnob

import os
import json
from tqdm import tqdm
import pickle
import re

import numpy as np
import scipy.io as sio
import cv2
import mne
import pandas as pd
from PIL import Image


class PreprocessingMAHNOBHCI(GenericVideoPreprocessing):
    def __init__(self, opts):
        super().__init__(opts)

        self.filename_pattern = config['filename_pattern']

        # The dictionary to store the session_id, subject_id, trial_id, data length, trimming range, etc.
        self.dataset_info = self.get_subject_trial_info()

        self.continuous_label_list = self.get_continuous_label()
        self.get_continuous_label_bool()
        self.get_eeg_bool()
        self.get_eeg_length()

        # The total of the sessions of the dataset.
        self.session_number = self.count_session()

        # Obtain the trimming range of the video for each trial.
        self.dataset_info['trim_range'] = self.get_video_trimming_range()

        # The config for the powerful openface.
        self.openface_config = config['openface_config']


        self.video_preprocessing()
        self.dataset_info['output_folder'] = self.get_output_folder_list()

        # Carry out the eeg preprocessing.
        # Filter the signal by bandpass filter ---> Filter the signal by notch filter
        # ---> independent component analysis ---> mean reference
        self.eeg_preprocessing()


        self.create_npy_for_frame()
        self.create_npy_for_success()
        self.create_npy_for_continuous_label()

        # Get the length (the amount of images in each folder)
        self.dataset_info['processed_length'] = self.get_processed_video_length()

        self.dataset_info['refined_processed_length'] = self.refine_processed_video_length()

        self.dataset_info['session_name'] = [directory.split(os.sep)[-1] for directory in self.dataset_info['output_folder']]

        self.create_npy_for_frame()
        self.create_npy_for_continuous_label()
        self.save_dataset_info()

    def save_dataset_info(self):
        r"""
        Save the dataset information for cross-operation-system consistency.
            It is required that the pre-processing is done in the same operation
            system, or even at the same computer. If not, the dataset info may be
            varied. (Because some operations like sorting are different in Windows
            and Linux, for an array having some equal elements, the sorting can be different,
            and cause a wrong dataset info.
        """
        pkl_filename = os.path.join(self.root_directory, "dataset_info.pkl")

        if not os.path.isfile(pkl_filename):
            with open(pkl_filename, 'wb') as pkl_file:
                pickle.dump(self.dataset_info, pkl_file)

    def create_npy_for_frame(self):

        for i, folder in tqdm(enumerate(range(self.session_number))):
            npy_directory = self.dataset_info['output_folder'][i]
            folder = self.dataset_info['processed_folder'][i] + "_aligned"
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_frame = os.path.join(npy_directory, "frame.npy")
            if not os.path.isfile(npy_filename_frame):
                frame_length = self.dataset_info['refined_processed_length'][i] * 16
                frame_list = get_filename_from_a_folder_given_extension(folder, ".jpg")[:frame_length]
                frame_matrix = np.zeros((frame_length, 120, 120, 3), dtype=np.uint8)

                for j, frame in enumerate(frame_list):
                    frame_matrix[j] = Image.open(frame)

                with open(npy_filename_frame, 'wb') as f:
                    np.save(f, frame_matrix)

    def create_npy_for_continuous_label(self):

        pointer = 0
        for i, folder in tqdm(enumerate(range(self.session_number))):
            if self.dataset_info['having_continuous_label'][i]:
                npy_directory = self.dataset_info['output_folder'][i]
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

    def get_output_folder_list(self):
        output_folder_list = []
        for folder in self.dataset_info['processed_folder']:
            output_folder = folder.replace("processed", "compacted_" + str(self.frame_size))
            output_folder_list.append(output_folder)
        return output_folder_list

    def count_session(self):
        r"""
        Count the sessions.
        :return: (int), the amount of sessions.
        """
        session_number = len(self.dataset_info["session_id"])
        return session_number

    def get_subject_trial_info(self):
        r"""
        :return:  the session_id, subject_id and trial_id of the dataset (dict).
        """
        directory = os.path.join(self.root_directory, "Sessions")
        session_id = np.asarray(
            sorted([idx for idx in os.listdir(directory) if os.path.isdir(os.path.join(directory, idx))], key=float),
            dtype=int)
        subject, trial = session_id // 130 + 1, session_id % 130

        dataset_info = {'session_id': session_id, 'subject_id': subject, 'trial_id': trial}
        return dataset_info

    def get_continuous_label_bool(self):
        continuous_label_mat_file = os.path.join(self.root_directory, "lable_continous_Mahnob.mat")
        self.dataset_info['having_continuous_label'] = np.zeros(len(self.dataset_info['session_id']), dtype=np.int32)
        mat_content = sio.loadmat(continuous_label_mat_file)
        sessions_having_continuous_label = mat_content['trials_included']
        unique_subject_index = [np.where(self.dataset_info['subject_id'] == n)[0][0] for n in np.unique(self.dataset_info['subject_id'])]

        for index in range(len(sessions_having_continuous_label)):
            subject, trial = sessions_having_continuous_label[index]
            start_idx = np.where(self.dataset_info['subject_id'] == subject)[0]
            offset = np.where(self.dataset_info['trial_id'][start_idx] == trial)[0][0]

            self.dataset_info['having_continuous_label'][start_idx[offset]] = 1

    def get_eeg_bool(self):
        r"""
        Some trials have no eeg recording. This function will indicate it.
        :return: (list), the binary to indicate the availability of eeg bdf file.
        """
        eeg_bool_list = []
        for folder in self.dataset_info['session_id']:
            flag = 0
            directory = os.path.join(self.root_directory, "Sessions", str(folder))
            for file in os.listdir(directory):
                if "emotion.bdf" in file:
                    flag = 1
            eeg_bool_list.append(flag)

        self.dataset_info['having_eeg'] =eeg_bool_list

    def get_eeg_length(self):

        eeg_length_list = []
        for folder in tqdm(self.dataset_info['session_id']):
            directory = os.path.join(self.root_directory, "Sessions", str(folder))
            length = 1e10
            for file in os.listdir(directory):
                if "emotion.bdf" in file:
                    eeg_file = os.path.join(directory, file)
                    raw_data = mne.io.read_raw_bdf(eeg_file, verbose=False)
                    length = raw_data.n_times - 256 * 60
            eeg_length_list.append(length)
        self.dataset_info['eeg_length'] = eeg_length_list

    def get_continuous_label(self):
        r"""
        :return: the continuous labels for each trial (dict).
        """

        label_file = os.path.join(self.root_directory, "lable_continous_Mahnob.mat")
        mat_content = sio.loadmat(label_file)
        annotation_cell = np.squeeze(mat_content['labels'])

        label_list = []
        for index in range(len(annotation_cell)):
            label_list.append(annotation_cell[index].T)
        return label_list

    def get_file_list_by_pattern(self, pattern):
        r"""
        :param pattern:  the regular expression of a file name (str). :return: the list of file names which satisfy
        the pattern (list), usually all the videos, or annotation files, etc.
        """
        dataset_info = self.dataset_info
        file_list = []

        # Iterate over the sessions.
        # For each session, find the file that matches the pattern, and store the file name in the python list.
        for index in range(self.session_number):
            directory = os.path.join(self.root_directory, "Sessions", str(dataset_info['session_id'][index]))

            # If the pattern is fixed.
            file_pattern = pattern

            # If the pattern contains variables, then fill the bracket accordingly.
            if "{}" in pattern:
                file_pattern = pattern.format(dataset_info['subject_id'][index], dataset_info['trial_id'][index])
                if "bdf" in pattern:
                    file_pattern = pattern.format(dataset_info['subject_id'][index],
                                                  dataset_info['trial_id'][index] // 2)

            # Carry out the regular expression matching
            reg_compile = re.compile(file_pattern)
            filename = [os.path.join(directory, file) for file in os.listdir(directory) if reg_compile.match(file)]
            if filename:
                file_list.append(filename[0])

        # If nothing found after the iteration, then the file should be single and located in the parent directory.
        if len(file_list) == 0:
            file_list = [os.path.join(self.root_directory, pattern)]

        return file_list

    def get_video_length(self):
        r"""
        :return: the length (frame count) of videos for each trial (dict).
        """
        video_list = self.get_file_list_by_pattern(self.filename_pattern['video'])
        lengths = np.zeros((len(video_list)), dtype=int)
        for index, video_file in enumerate(video_list):
            video = cv2.VideoCapture(video_file)
            lengths[index] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return lengths

    def get_video_trimming_range(self):
        r"""
        Get the trimming intervial of the videos.
        :return: (ndarray), the ranges to guide the video trimming.
        """
        intermediate_range_pkl_file = os.path.join(self.root_directory, "intermediate_range.pkl")
        if not os.path.isfile(intermediate_range_pkl_file):
            record_file_list = self.get_file_list_by_pattern(self.filename_pattern['timestamp'])
            ranges = read_start_end_from_mahnob_tsv(record_file_list)
            with open(intermediate_range_pkl_file, 'wb') as f:
                pickle.dump(ranges, f)
        else:
            with open(intermediate_range_pkl_file, 'rb') as f:
                ranges = pickle.load(f)
        return ranges

    def get_processed_video_length(self):
        r"""
        After changing the fps and processing the video using OpenFace, the video length
            has been changed, therefore we have to get the length again.
        :return: (list) the length list of the processed video in the format of static images.
        """

        length_list = []

        if not os.path.isfile(os.path.join(self.root_directory, 'processed_video_length.pkl')):
            for folder in tqdm(self.dataset_info['processed_folder']):
                csv_file = folder + ".csv"
                length = len(pd.read_csv(csv_file))
                length_list.append(length)
            save_pkl_file(self.root_directory, 'processed_video_length.pkl', length_list)
        else:
            length_list = load_single_pkl(os.path.join(self.root_directory, 'processed_video_length'))

        return length_list

    def refine_processed_video_length(self):
        r"""
        Some trials have continuous labels while others are not. This function will refine the ones
            having continuous labels, so that the length will be corresponding with the continuous
            label length and frame-to-label ratio.
        :return: (list) the length list of the processed video in the format of static images.
        """
        length_list = []
        pointer = 0
        for index in tqdm(range(len(self.dataset_info['trial_id']))):
            video_length = self.dataset_info['processed_length'][index]
            eeg_length = self.dataset_info['eeg_length'][index]
            if self.dataset_info['having_continuous_label'][index]:
                length = len(self.continuous_label_list[pointer])
                fine_length = min(length, video_length // 16, eeg_length // 64)
                pointer += 1
            else:
                fine_length = min(video_length // 16, eeg_length // 64)

            length_list.append(fine_length)

        return length_list

    def eeg_preprocessing(self):
        r"""
        :return: Carry out the eeg preprocessing.
        """
        eeg_bdf_list = self.get_file_list_by_pattern(self.filename_pattern['eeg'])
        pointer = 0
        for index in tqdm(range(len(self.dataset_info['session_id']))):

            if self.dataset_info['having_eeg'][index]:
                output_directory = self.dataset_info['output_folder'][index]
                os.makedirs(output_directory, exist_ok=True)
                output_file = os.path.join(output_directory, "eeg_raw.npy")

                if not os.path.isfile(output_file):
                    eeg_handler = EegMahnob(eeg_bdf_list[pointer], buffer=5)
                    eeg_data = eeg_handler.filtered_data
                    np.save(output_file, eeg_data)

                pointer += 1

    def video_preprocessing(self):
        r"""
        :return: carry out the preprocessing for videos.
        """
        video_list = self.get_file_list_by_pattern(self.filename_pattern['video'])

        # Pick only the annotated clips from a complete video.
        video_list = combine_annotated_clips(
            video_list, self.dataset_info['trim_range'], direct_copy=False, visualize=False)

        # Change the fps of the video to a integer.
        video_list = change_video_fps(video_list, self.target_fps)

        output_directory = os.path.join(self.root_directory, self.openface_output_folder)
        os.makedirs(output_directory, exist_ok=True)

        # Extract facial landmark, warp, crop, and save each frame.
        openface = OpenFaceController(openface_path=self.openface_config['openface_directory'],
                                      output_directory=output_directory)
        video_list = openface.process_video_list(video_list=video_list, dataset_info=self.dataset_info, **self.openface_config)

        # Save the static folders
        self.dataset_info['processed_folder'] = video_list


if __name__ == "__main__":
    with open("config_mahnob") as config_file:
        config = json.load(config_file)

    pre = PreprocessingMAHNOBHCI(config)