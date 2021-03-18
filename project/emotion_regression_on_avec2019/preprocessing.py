from base.preprocessing import GenericVideoPreprocessing
from base.utils import get_filename_from_a_folder_given_extension, get_filename_from_full_path, get_video_length, save_pkl_file, copy_file
from base.video import OpenFaceController

import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image


class PreprocessingAVEC2019(GenericVideoPreprocessing):
    def __init__(self, opts):
        super().__init__(opts)
        self.partition_list = opts['partition_list']
        self.country_list = opts['country_list']

        self.establish_dataset_info()

        self.video_preprocessing()
        self.feature_preprocessing()
        self.label_preprocessing()

        self.create_npy_for_frame()
        self.create_npy_for_success()
        self.create_npy_for_continuous_label()

    def video_preprocessing(self):
        video_list = []
        param = self.openface_config
        subject_id = 1
        print("\nProcessing videos...")
        for partition_name in self.partition_list:
            if partition_name != "Test":
                directory = os.path.join(self.root_directory, self.raw_data_folder, partition_name)
                label_list = get_filename_from_a_folder_given_extension(directory, "csv")
                for _ in tqdm(label_list):
                    new_name = "P" + str(subject_id).zfill(2) + "-T01"
                    raw_name = self.dataset_info['partition_country_to_participant_trial_map'][new_name]

                    raw_fullname = os.path.join(self.root_directory, self.raw_data_folder, partition_name, raw_name + '.avi')

                    output_directory = os.path.join(self.root_directory, self.openface_output_folder)
                    os.makedirs(output_directory, exist_ok=True)

                    openface = OpenFaceController(openface_path=self.openface_config['openface_directory'], output_directory=output_directory)
                    _ = openface.process_video(
                        input_filename=raw_fullname, output_filename=new_name, **self.openface_config)

                    subject_id += 1

    def feature_preprocessing(self):
        directory = os.path.join(self.root_directory, self.openface_output_folder)
        csv_list = get_filename_from_a_folder_given_extension(directory, "T01.csv")
        print("\nProcessing features...")
        for csv_file in tqdm(csv_list):
            output_csv_success_indices_fullname = csv_file.split(".csv")[0] + "_success.csv"
            if not os.path.isfile(output_csv_success_indices_fullname):
                frame_indices = pd.read_csv(csv_file,
                                            skipinitialspace=True, usecols=["success"],
                                            index_col=False).values.squeeze()

                data_frame = pd.DataFrame(frame_indices, columns=["success"])
                data_frame.to_csv(output_csv_success_indices_fullname, index=False)

    def label_preprocessing(self):
        print("\nProcessing continuous labels...")
        for partition_name in self.partition_list:
            directory = os.path.join(self.root_directory, self.raw_data_folder, partition_name)
            label_list = get_filename_from_a_folder_given_extension(directory, ".csv")
            for label_fullname in tqdm(label_list):
                raw_filename = get_filename_from_full_path(label_fullname).split(".csv")[0]
                new_filename = self.dataset_info['participant_trial_to_partition_country_map'][raw_filename] + "_continuous_label.csv"
                new_fullname = os.path.join(self.root_directory, self.openface_output_folder, new_filename)
                if not os.path.isfile(new_fullname):
                    copy_file(label_fullname, new_fullname)

    def create_npy_for_continuous_label(self):
        directory = os.path.join(self.root_directory, self.openface_output_folder)
        csv_list = get_filename_from_a_folder_given_extension(directory, "_continuous_label.csv")
        print("\nCreating npy files for videos...")
        for index, csv_file in tqdm(enumerate(csv_list)):
            npy_directory = os.path.join(self.root_directory, self.npy_folder, csv_file.split(os.sep)[-1].split("_continuous_label.csv")[0])
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_continuous_label = os.path.join(npy_directory, "continuous_label.npy")

            cols = [emotion.lower() for emotion in self.emotion_dimension]

            if not os.path.isfile(npy_filename_continuous_label):
                continuous_label = pd.read_csv(csv_file, sep=";",
                                            skipinitialspace=True, usecols=cols,
                                            index_col=False).values.squeeze()
                continuous_label = continuous_label[:self.dataset_info['frame_count'][index] // 5]

                with open(npy_filename_continuous_label, 'wb') as f:
                    np.save(f, continuous_label)

    def create_npy_for_success(self):
        directory = os.path.join(self.root_directory, self.openface_output_folder)
        csv_list = get_filename_from_a_folder_given_extension(directory, "T01.csv")
        print("\nCreating npy files for features...")
        for index, csv_file in tqdm(enumerate(csv_list)):
            npy_directory = os.path.join(self.root_directory, self.npy_folder, csv_file.split(os.sep)[-1].split(".csv")[0])
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_success = os.path.join(npy_directory, "success.npy")
            if not os.path.isfile(npy_filename_success):
                success_indices = pd.read_csv(csv_file,
                                            skipinitialspace=True, usecols=["success"],
                                            index_col=False).values.squeeze()
                success_indices = success_indices[:self.dataset_info['frame_count'][index]]

                with open(npy_filename_success, 'wb') as f:
                    np.save(f, success_indices)

    def create_npy_for_frame(self):
        directory = os.path.join(self.root_directory, self.openface_output_folder)
        trials_folder = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
        print("\nCreating npy files for videos...")
        for i, folder in tqdm(enumerate(trials_folder)):
            npy_directory = os.path.join(self.root_directory, self.npy_folder, folder.split(os.sep)[-1].split("_aligned")[0])
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_frame = os.path.join(npy_directory, "frame.npy")
            if not os.path.isfile(npy_filename_frame):
                frame_list = get_filename_from_a_folder_given_extension(folder, ".jpg")[:self.dataset_info['frame_count'][i]]
                frame_length = len(frame_list)
                frame_matrix = np.zeros((frame_length, self.frame_size, self.frame_size, 3), dtype=np.uint8)

                for j, frame in enumerate(frame_list):
                    frame_matrix[j] = Image.open(frame)

                with open(npy_filename_frame, 'wb') as f:
                    np.save(f, frame_matrix)

    def establish_dataset_info(self):
        raw_directory = os.path.join(self.root_directory, self.raw_data_folder)
        participant_trial_to_partition_country_map, partition_country_to_participant_trial_map = {}, {}
        subject_id = 1
        for partition in self.partition_list:
            directory = os.path.join(raw_directory, partition)
            # video_list = get_filename_from_a_folder_given_extension(directory, "avi")
            label_list = get_filename_from_a_folder_given_extension(directory, "csv")
            for raw_fullname in label_list:
                raw_filename = get_filename_from_full_path(raw_fullname).split('.csv')[0]
                raw_video_fullname = raw_fullname.split(".csv")[0] + ".avi"
                new_filename = "P" + str(subject_id).zfill(2) + "-T01"
                participant_trial_to_partition_country_map[raw_filename] = new_filename
                partition_country_to_participant_trial_map[new_filename] = raw_filename
                this_partition = raw_filename.split("_")[0]
                country = raw_filename.split("_")[1]
                self.dataset_info['subject_id'].append(subject_id)
                self.dataset_info['trial_id'].append(1)
                self.dataset_info['frame_count'].append(get_video_length(raw_video_fullname) // 5 * 5)
                self.dataset_info['partition'].append(this_partition)
                self.dataset_info['country'].append(country)
                self.dataset_info['have_continuous_label'].append(1)
                subject_id += 1
        self.dataset_info['participant_trial_to_partition_country_map'] = participant_trial_to_partition_country_map
        self.dataset_info['partition_country_to_participant_trial_map'] = partition_country_to_participant_trial_map
        self.dataset_info['subject_id'] = np.asarray(self.dataset_info['subject_id'])
        self.dataset_info['trial_id'] = np.asarray(self.dataset_info['trial_id'])
        self.dataset_info['have_continuous_label'] = np.asarray(self.dataset_info['have_continuous_label'])
        if not os.path.isfile(os.path.join(self.root_directory, 'dataset_info.pkl')):
            save_pkl_file(self.root_directory, 'dataset_info.pkl', self.dataset_info)

    @staticmethod
    def init_dataset_info():
        dataset_info = {
            "participant_trial_to_partition_country_map": [],
            "partition_country_to_participant_trial_map": [],
            "subject_id": [],
            "trial_id": [],
            "frame_count": [],
            "partition": [],
            "country": [],
            "have_continuous_label": [],
        }

        return dataset_info


if __name__ == "__main__":
    with open("config_avec2019") as config_file:
        config = json.load(config_file)

    pre = PreprocessingAVEC2019(config)