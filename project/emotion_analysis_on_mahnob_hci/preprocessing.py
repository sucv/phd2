from base.preprocessing import GenericVideoPreprocessing
from base.utils import get_filename_from_a_folder_given_extension, save_pkl_file
from base.video import OpenFaceController
from base.utils import load_single_pkl
from base.video import combine_annotated_clips, change_video_fps
from base.eeg import azim_proj
from project.emotion_analysis_on_mahnob_hci.utils import read_start_end_from_mahnob_tsv, number_to_emotion_tag_dict, emotion_tag_to_arousal_class, emotion_tag_to_valence_class, arousal_class_to_number, valence_class_to_number
from project.emotion_analysis_on_mahnob_hci.eeg import EegMahnob


import os
import xml.etree.ElementTree as et
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
    def __init__(self, config):
        super().__init__(config)

        self.filename_pattern = config['filename_pattern']

        # The dictionary to store the session_id, subject_id, trial_id, data length, trimming range, etc.
        self.dataset_info = self.get_subject_trial_info()

        self.continuous_label_list = self.get_continuous_label()
        self.get_continuous_label_bool()
        self.get_eeg_bool()
        self.get_eeg_length()
        self.eeg_electrode_list = self.get_eeg_electrode_list()
        self.eeg_electrode_3d_coordinates = self.get_eeg_electrode_3d_coordinates(self.eeg_electrode_list)
        self.eeg_electrode_2d_coordinates = self.get_eeg_electrode_2d_coordinates(self.eeg_electrode_3d_coordinates)

        self.eeg_feature_extraction = self.config['eeg_feature_extraction']
        self.eeg_feature_list = self.config['eeg_feature_list']

        # The total of the sessions of the dataset.
        self.session_number = self.count_session()

        # Obtain the trimming range of the video for each trial.
        self.dataset_info['trim_range'] = self.get_video_trimming_range()

        # The config for the powerful openface.
        self.openface_config = config['openface_config']


        self.video_preprocessing()
        self.dataset_info['npy_output_folder'] = self.get_npy_output_folder_list()

        # Carry out the eeg preprocessing.
        # Filter the signal by bandpass filter ---> Filter the signal by notch filter
        # ---> independent component analysis ---> mean reference
        self.eeg_preprocessing()


        # Get the length (the amount of images in each folder)
        self.dataset_info['processed_length'] = self.get_processed_video_length()

        self.dataset_info['refined_processed_length'] = self.refine_processed_video_length()

        self.dataset_info['session_name'] = [directory.split(os.sep)[-1] for directory in self.dataset_info['npy_output_folder']]

        self.create_npy_for_frame()
        self.create_npy_for_continuous_label()

        self.class_label_preprocessing()
        self.save_dataset_info()

    @staticmethod
    def get_eeg_electrode_2d_coordinates(electrodes):
        elec_coordinate_list = []
        for electrode in electrodes:
            elec_coordinate = azim_proj(electrode)
            elec_coordinate_list.append(elec_coordinate)

        elec_coordinate_list = np.asarray(elec_coordinate_list)
        return elec_coordinate_list

    @staticmethod
    def get_eeg_electrode_3d_coordinates(electrodes):
        import eeg_positions
        elec_coordinates = eeg_positions.get_elec_coords(system="1010", elec_names=electrodes, dim="3d")
        elec_coordinates = elec_coordinates.values[:, 1:]
        return elec_coordinates


    @staticmethod
    def get_eeg_electrode_list():
        electrode_list = [
            "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7",
            "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
            "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"]
        return electrode_list

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

        for i, folder in tqdm(enumerate(range(self.session_number)), total=self.session_number):
            npy_directory = self.dataset_info['npy_output_folder'][i]
            folder = self.dataset_info['processed_folder'][i] + "_aligned"
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_frame = os.path.join(npy_directory, "frame.npy")
            if not os.path.isfile(npy_filename_frame):
                frame_length = self.dataset_info['refined_processed_length'][i] * 16
                frame_list = get_filename_from_a_folder_given_extension(folder, ".jpg")[:frame_length]
                frame_matrix = np.zeros((frame_length, self.frame_size, self.frame_size, 3), dtype=np.uint8)

                for j, frame in enumerate(frame_list):
                    frame_matrix[j] = Image.open(frame)

                with open(npy_filename_frame, 'wb') as f:
                    np.save(f, frame_matrix)

    def create_npy_for_continuous_label(self):

        pointer = 0
        for i, folder in tqdm(enumerate(range(self.session_number)), total=self.session_number):
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

    def get_npy_output_folder_list(self):
        npy_output_folder_list = []
        for folder in self.dataset_info['processed_folder']:
            npy_output_folder = folder.replace("processed_" + str(self.frame_size), "compacted_" + str(self.frame_size))
            npy_output_folder_list.append(npy_output_folder)
        return npy_output_folder_list

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
            if len(start_idx) != 0:
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
                output_directory = self.dataset_info['npy_output_folder'][index]
                os.makedirs(output_directory, exist_ok=True)
                output_eeg_raw_file = os.path.join(output_directory, "eeg_raw.npy")
                output_eeg_psd_file = os.path.join(output_directory, "eeg_psd.npy")
                output_eeg_image_file = os.path.join(output_directory, "eeg_image.npy")

                if self.eeg_feature_extraction:
                    # In addition to the EEG data relevant to the continuous labels, an extra length equalling 5s is sampled.
                    # So that it can ensure that the length of each EEG sample will not lack one or to data point.
                    eeg_handler = EegMahnob(
                        eeg_bdf_list[pointer], buffer=5, electrode_2d_pos=self.eeg_electrode_2d_coordinates,
                        eeg_image_size=self.config['crop_size'], eeg_feature_list=self.eeg_feature_list)

                    if "raw_data" in self.eeg_feature_list:
                        eeg_data = eeg_handler.extracted_data['raw_data']
                        if not os.path.isfile(output_eeg_raw_file):
                            np.save(output_eeg_raw_file, eeg_data)

                    if "psd" in self.eeg_feature_list:
                        eeg_psd = eeg_handler.extracted_data['psd']
                        if not os.path.isfile(output_eeg_psd_file):
                            np.save(output_eeg_psd_file, eeg_psd)

                    if "eeg_image" in self.eeg_feature_list:
                        eeg_image = eeg_handler.extracted_data['eeg_image']
                        if not os.path.isfile(output_eeg_image_file):
                            np.save(output_eeg_image_file, eeg_image)

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

    def class_label_preprocessing(self):
        xml_list = self.get_file_list_by_pattern(self.filename_pattern['session_log'])
        emotion_dict = {}
        for index, xml_file in tqdm(enumerate(xml_list), total=len(xml_list)):

            if self.dataset_info['having_eeg'][index]:
                xml_file = et.parse(xml_file).getroot()
                felt_emotion = xml_file.find('.').attrib['feltEmo']
                felt_arousal = xml_file.find('.').attrib['feltArsl']
                felt_valence = xml_file.find('.').attrib['feltVlnc']

                intermediate_dict = {
                    self.dataset_info['session_name'][index]:
                        {
                            "Arousal": arousal_class_to_number[emotion_tag_to_arousal_class[number_to_emotion_tag_dict[felt_emotion]]],
                            "Valence": valence_class_to_number[emotion_tag_to_valence_class[number_to_emotion_tag_dict[felt_emotion]]]
                        }
                    }
            else:
                intermediate_dict = {self.dataset_info['session_name'][index]: []}
            emotion_dict.update(intermediate_dict)

        save_pkl_file(self.root_directory, "class_label.pkl", emotion_dict)


if __name__ == "__main__":
    from project.emotion_analysis_on_mahnob_hci.configs import config_mahnob as config
    pre = PreprocessingMAHNOBHCI(config)