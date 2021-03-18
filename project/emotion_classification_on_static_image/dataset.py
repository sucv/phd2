import h5py
import os
from operator import itemgetter

from PIL import Image
import numpy as np
import random
import pandas as pd
import skvideo.io
import cv2
from utils import transforms3D


# import torch.utils.data
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils.helper import load_single_pkl, load_pkl_file, dict_combine



class EmotionalStaticImgClassificationDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.transform = transform
        self.dataset = data_list
        self.targets = self.get_targets()

    def get_targets(self):
        targets = [int(target[0]) for target in self.dataset]
        return targets

    def __getitem__(self, index):
        label, img_filename = self.dataset[index]

        label = torch.LongTensor([int(label)])
        img = self.transform(Image.open(img_filename))

        return img, label

    def __len__(self):
        return len(self.dataset)


class EmotionalDataset(Dataset):
    r"""
    The dataset class to read the video clips and their associate continuous labels.
    """
    def __init__(self, data_to_load, time_depth):

        # The video list to read.
        self.video_list = data_to_load['frame']

        # The associated continuous label to read.
        self.continuous_label_list = data_to_load['continuous_label']

        # The length of a sample.
        self.time_depth = time_depth

        # Load.
        self.video = self.load_frame()
        self.continuous_label = self.load_continuous_label()

    def load_frame(self):
        # Load the video files then stack to a single matrix.
        video = np.vstack([skvideo.io.vread(video_file) for video_file in self.video_list])

        # Convert to a float tensor.
        video = torch.from_numpy(video).to(dtype=torch.float32) / 255.

        # Permute the dimension so that it is now in the order of time_depth x channel x width x height.
        video = video.permute(0, 3, 1, 2)
        return video

    def load_continuous_label(self):
        r"""
        Load the associated continuous labels.
        :return: the continuous labels with an extra dimension to indicating from which
            file the labels come. This dimension will help the output shape restoring so that
            the metrics for model training can be correctly calculated on each session, instead of
            on each sample.
        """

        continuous_labels = []
        for index, h5file in enumerate(self.continuous_label_list):
            with h5py.File(h5file, "r") as f:
                continuous_label = f['ndarray'][()]

                # Later we will separate the extra dimension from the label array.
                continuous_labels.append(continuous_label)

        continuous_label = torch.from_numpy(np.vstack(continuous_labels)).type(torch.float32)
        return continuous_label

    def __getitem__(self, idx):

        start_idx = idx * self.time_depth
        end_idx = (idx + 1) * self.time_depth

        # Here we also output the file index from which the sample come,
        # for later restoration as mentioned above.
        sample_indicator = idx
        sampled_video = self.video[start_idx:end_idx, :, :, :]
        sampled_continuous_label = self.continuous_label[start_idx:end_idx, :]
        return sampled_video, sampled_continuous_label, sample_indicator

    def __len__(self):
        return np.shape(self.continuous_label)[0] // self.time_depth


class EmotionalFramewiseDataset(Dataset):
    r"""
    [Deprecated, because load video frame-by-frame is way too slow.]
    A class to load video in consecutive jpg format.
        Specifically, time_depth jpgs will form a window, with a stride of step_size.
    This class is dependent to the filename list of the jpg files from the Arranger.
    """

    def __init__(self, frame_list, ndarray, config, transform=None):

        self.time_depth = config['time_depth']
        self.step_size = config['step_size']
        self.data_type = config['data_type']

        # The filename list of the jpg files.
        self.frame_list = frame_list
        self.tensors = self.init_tensor()

        # The continuous labels corresponding to the frame_list.
        self.ndarray = ndarray
        self.length = self.get_length()
        self.transform = transform

        self.slice = self.generate_indices("slice")
        self.xrange = self.generate_indices("xrange")

        # Load the jpg files and continuous label with a specific window and stride.
        self.data = self.get_data()
        self.continuous_label = self.get_continuous_label()

    def init_tensor(self):
        return torch.Tensor(len(self.frame_list), 3, 224, 224)

    @staticmethod
    def load_image(index, frame_list, tensors, transform):
        path = frame_list[index]
        image = Image.open(path)
        image = transform(image)
        tensors[index, :, :, :] = image
        print(index)

    def get_data(self):
        r"""
        Load the data. It is currently assumed that the data are frames. In the future
            maybe the action unit will be added.
        :return: (dict), the dictionary containing the data under a specific key.
        """
        data = {key: [] for key in self.data_type}
        if "frame" in data:
            images = self.load_frame()
            data.update({'frame': images})
        return data

    def load_frame(self):
        r"""
        Load the jpg files using PIL API.
        :return: (tensor), the data as a tensor.
        """
        images = torch.Tensor(len(self.frame_list), 3, 224, 224)

        for index, path in enumerate(self.frame_list):
            image = Image.open(path)
            image = self.transform(image)
            images[index, :, :, :] = image
            print(index)

        return images

    def get_continuous_label(self):
        r"""
        Get the continuous label. It is currently assumed that the ndarray contain only
            the continuous label. In the future maybe the action unit will be added.
        :return: (ndarray), the continuous label.
        """
        continuous_label = self.ndarray
        return continuous_label

    def get_length(self):
        r"""
        Get the count of how many frames are to be processed.
        :return: (int), the count of the frames.
        """
        return len(self.frame_list)

    def generate_indices(self, type_of_indices):
        r"""
        Generate the indices to guide the windowing with stride.
        :param type_of_indices: (str), the types of the indices to be generated. Can be
            a slice object for list, or xrange object for ndarray.
        :return: (list), the list of indices.
        """
        window_count = int((self.length / self.step_size) - 1)
        indices_list = []
        for window in range(window_count):
            start = window * self.step_size
            end = start + self.time_depth
            if end > self.length:
                start = self.length - self.time_depth
                end = self.length
            if type_of_indices == "slice":
                indices_list.append(slice(start, end))
            elif type_of_indices == "xrange":
                indices_list.append(np.arange(start, end))

        return indices_list

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, idx):
        r"""
        Sample one window at a time.
        :param idx: (int), the index to sample.
        :return: the samples.
        """
        frame = self.data['frame'][self.slice[idx], :, :, :]
        target = self.continuous_label[self.slice[idx], :]

        return frame, target



class DataArrangerSemaine:
    r"""
    This class generates a list for the dataset class to load. It is used to overcome
        the inconvenience of data loading of videos in jpg format.
    """

    def __init__(self, config, session_id_of_n_fold):
        self.root_directory = config['root_directory']
        self.data_folder = config['data_folder']
        self.emotion_dimension = config['emotion_dimension']
        self.session_id_of_n_fold = session_id_of_n_fold

        # Determine how many frames are to be loaded in memory. Wisely set it according
        # to the specs of the PC.
        self.frame_number_to_compact = config['frame_number_to_compact']

        # # For every downsampling_interval frames would the Arranger load one for training.
        # self.downsampling_interval = downsampling_interval

        self.time_depth = config['time_depth']
        self.step_size = config['step_size']

        self.dataset_info = self.get_dataset_info()

    def get_dataset_info(self):
        r"""
        Read the dataset info generated by the proprocessing.
        :return: (dict), the dataset info.
        """
        dataset_info = load_single_pkl(self.root_directory, "dataset_info")
        return dataset_info

    def get_session_count(self):
        r"""
        Count how many sessions having the continuous label are to be processed.
        :return: (int), the count of the sessions having continuous label.
        """
        session_count = np.sum(self.dataset_info['feeltrace_bool'])
        return session_count

    def get_session_having_feeltrace(self):
        r"""
        Select only the sessions having continuous labels.
        :return: (ndarray), the indices of the selected sessions.
        """
        sessions_having_feeltrace = np.where(self.dataset_info['feeltrace_bool'] == 1)[0]

        return sessions_having_feeltrace

    def get_subject_id_having_feeltrace(self):

        sessions_having_feeltrace = self.get_session_having_feeltrace()
        return np.unique(self.dataset_info['subject_id'][sessions_having_feeltrace])


    def get_session_string_list(self):
        r"""
        Generate the filename of a specific session.
        :return: (list), the filename list.
        """
        session_string_list = []

        sessions_having_feeltrace = self.get_session_having_feeltrace()
        for session in sessions_having_feeltrace:
            subject_id = self.dataset_info['subject_id'][session]
            trial_id = self.dataset_info['trial_id'][session]
            session_string = "P{}-T{}".format(subject_id, trial_id)
            session_string_list.append(session_string)

        return session_string_list

    def get_file_list(self, suffix):
        r"""
        Get the filename list containing a specific suffix.
        :param suffix: (str), the suffix to include.
        :return: (list), the list of selected filename.
        """
        session_string_list = self.get_session_string_list()
        file_list = [session_string + suffix for session_string in session_string_list]
        return file_list

    def get_frame_jpg_list(self, session_list):
        r"""
        Get the filename list of the jpg files for a session.
        :param session_list: (list), the sessions to load.
        :return: (list), the list of jpg filename.
        """
        absolute_frame_jpg_list = []
        for session in session_list:

            frame_folder = self.get_file_list("_aligned")[session]
            frame_folder_directory = os.path.join(self.root_directory, self.data_folder, frame_folder)
            frame_jpg_list = sorted(os.listdir(frame_folder_directory))
            absolute_frame_jpg_list.append([os.path.join(
                self.root_directory, self.data_folder, frame_folder, filename) for filename in frame_jpg_list])
        return absolute_frame_jpg_list

    def get_ndarraies(self, session_list):
        r"""
        Get the continuous label corresponding to the selected frames.
        :param session_list: (list), the sessions to load.
        :return: (ndarray), the continuous label of the selected frames.
        """
        continuous_label_list = []
        for session in session_list:
            continuous_label_csv_filename = self.get_file_list("_continuous_label.csv")[session]
            continuous_label_csv_directory = os.path.join(self.root_directory, self.data_folder,
                                                          continuous_label_csv_filename)
            continuous_label_list.append(pd.read_csv(continuous_label_csv_directory).to_numpy())
        return continuous_label_list

    def get_success_indices(self, session_list):
        r"""
        Get the indices of the frames with the face detected.
        :param session_list: (list), the sessions to load.
        :return: (ndarray), the success indices of the selected frames.
        """
        success_indices_list = []
        for session in session_list:
            success_indices_csv_filename = self.get_file_list("_success_indices.csv")[session]
            success_indices_csv_directory = os.path.join(self.root_directory, self.data_folder,
                                                         success_indices_csv_filename)
            success_indices_list.append(pd.read_csv(success_indices_csv_directory).to_numpy()[:, 0])
        return success_indices_list

    @staticmethod
    def generate_downsampling_indices(length_list, downsampling_interval, random_interval=False):
        r"""
        Downsample the video by selecting one frame for every downsampling_interval
            frames. If random_interval is enabled, the interval will range in (0, downsampling_interval)
        :param length_list: (list), the list recording the frame number of every video.
        :param downsampling_interval: (int), the stepsize of the downsampling indices.
        :param random_interval: (bool), if enabled, random variations
            within [0, downsampling_interval] will be applied on the downsampling indices.
        :return: (list), the downsampling_indices_list, which will later be applied to the frames
            having detected faces.
        """

        downsampling_indices_list = []

        # Iterate the length_list
        for length in length_list:

            # Count the pointers, discard the last one which usually exceed the available length.
            pointer_number = length // downsampling_interval

            # Calculate the non-consecutive pointers with a fixed interval.
            downsampling_indices = np.arange(0, length, downsampling_interval)[:pointer_number]

            # If random_interval is enabled, then apply random variations on the
            # pointer positions.
            if random_interval:
                random_indices = np.random.randint(downsampling_interval, size=pointer_number)
                downsampling_indices += random_indices

            # Record the downsampling indices.
            downsampling_indices_list.append(downsampling_indices)
        return downsampling_indices_list

    def generate_data_filename_and_label_array(
            self, session_id_of_subjects, downsampling_interval=1):
        r"""
        Finally, generate a dictionary to save the successful frame filenames, and the corresponding
            ndarraies.
        :param session_id_of_subjects: (list), the session to load.
        :param downsampling_interval: (int), for every downsampling_interval steps will the sample perform.
        :return: (dict), the dictionary containing the data filenames and their continuous labels.
        """
        session_having_feeltrace = self.get_session_having_feeltrace()
        subject_id_for_sessions_having_feeltrace = self.dataset_info['subject_id'][session_having_feeltrace]
        session_list = np.hstack(session_id_of_subjects)

        # Get the frame filenames, their associated ndarraies and success indices.
        # Note that extra feature in addition to facial frames can be stored in the ndarraies,
        # i.e., the continuous_label_list currently. In this case, the argument continuous_label_list should
        # include the extra features (e.g., the action unit) from earlier processing (i.e., self.get_continuous_label())
        frame_jpg_list = self.get_frame_jpg_list(session_list)
        ndarray_list = self.get_ndarraies(session_list)
        success_indices_list = self.get_success_indices(session_list)

        # Obtain the length list for each video (consider only the frames with faces detected).
        length_list = [len(success_indices) for success_indices in success_indices_list]

        # Generate the pointers for downsampling.
        downsampling_indices_list = self.generate_downsampling_indices(
            length_list, downsampling_interval)

        # Apply the downsampling indices on the success indices.
        success_indices_list = [success_indices[downsampling_indices]
                                for success_indices, downsampling_indices in
                                zip(success_indices_list, downsampling_indices_list)]

        # Apply the success_indices on their associated frames and ndarraies.
        success_frame_jpg_list = [list(itemgetter(*success_indices)(frame_jpg_list))
                                  for success_indices, frame_jpg_list in
                                  zip(success_indices_list, frame_jpg_list)]
        success_ndarray_list = [continuous_label[success_indices]
                                for success_indices, continuous_label in
                                zip(success_indices_list, ndarray_list)]

        # Initialize an empty list to record.
        generated_list = []

        sliced_length_dict = {str(key): [] for key in subject_id_for_sessions_having_feeltrace}
        # Synchronously process the frame filename list and ndarray list
        for success_frame_jpgs, success_ndarraies, subject_id in zip(
                success_frame_jpg_list, success_ndarray_list, subject_id_for_sessions_having_feeltrace):

            length = len(success_frame_jpgs)
            sliced_length_list = []
            generated_data_and_file = {'frame': success_frame_jpgs, 'ndarray': success_ndarraies}
            # Discard very small batch which has less then self.time_depth lines. Such
            # batches are usually the residual of big sessions.
            if length >= self.time_depth:

                # Calculate the window number given the sampling size and strides.
                window_count = (length - self.time_depth) // self.step_size + 1

                # The sampling indices are save as a list of slices. The slices are used to ensure
                # the temporal continuity for every self.time_depth lines.
                slice_list = [slice(start * self.step_size, start * self.step_size + self.time_depth) for start in
                              range(window_count)]

                # Initialize the dictionary to record the sliced list. If self.step_size is smaller than
                # self.time_depth, then the sliced dictionary will be larger than its precedent.
                generated_data_and_file_sliced = {key: [] for key in ["frame", "ndarray"]}

                # Combine lines of each slice to the sliced dictionary.
                for slicing in slice_list:
                    intermediate_dict = {key: value[slicing] for key, value in generated_data_and_file.items()}
                    generated_data_and_file_sliced = dict_combine(generated_data_and_file_sliced, intermediate_dict)

                # Stack the elementary lists into a single one for later convenience.
                generated_data_and_file_sliced = {
                    key: (np.hstack(value).tolist() if key == 'frame' else np.vstack(value)) for key, value in
                    generated_data_and_file_sliced.items()}

                generated_list.append(generated_data_and_file_sliced)
                sliced_length_dict[str(subject_id)].append((window_count - 1) * self.step_size + self.time_depth)

        compacted_dict = {}

        subject_id_list = self.get_subject_id_having_feeltrace()
        subject_clip_sample_info = {}

        for index, session_id_of_a_subject in enumerate(session_id_of_subjects):

            relative_session_id = -1
            subject_id = subject_id_list[index]

            # Process the things for one subject at a time.
            data_and_file_for_one_subject = list(itemgetter(*session_id_of_a_subject)(generated_list))

            compacted_intermediate_dict = {key: [] for key in ["frame", "ndarray"]}
            for data_and_file_for_one_session in data_and_file_for_one_subject:
                compacted_intermediate_dict = dict_combine(compacted_intermediate_dict, data_and_file_for_one_session)

            # Stack the elementary lists into a single one for later convenience.
            compacted_intermediate_dict = {
                key: (np.hstack(value).tolist() if key == 'frame' else np.vstack(value)) for key, value in
                compacted_intermediate_dict.items()}

            # This variable (sample_id) is crucial. It records the indices relative to each session of a subject.
            # A window contain time_depth frames, they constitute a sample.
            # Later, when we obtain the output from the network, we have to place that output
            # segment to the corresponding position of a session-wise long array.
            # This variable can help to determine the correct positions for each output segment.
            # It records the indices relative to each session of a subject. So that even if a clip can occasionally
            # have data from two sessions (this happens when a clip contains more than one samples),
            # we can still retrieve the sample-wise affiliation according to this variable.
            sample_id = -1

            break_flag = False
            clip_list = []

            # Split the data of a given session to several batches, each with a smaller size so that
            # they can be loaded by a commercial PC or server flexibly.

            clip_info_dict = {}
            for clip_id in range(len(compacted_intermediate_dict['frame']) // self.frame_number_to_compact + 1):

                if break_flag:
                    break

                # Determine the batch window. Note that the window amount must be
                # the multiplier of self.time_depth
                start = clip_id * self.frame_number_to_compact
                end = (clip_id + 1) * self.frame_number_to_compact
                if len(compacted_intermediate_dict['frame']) - end <= self.frame_number_to_compact * 0.2:
                    end = len(compacted_intermediate_dict['frame'])
                    break_flag = True

                # Get the frame and continuous label for a clip. Meanwhile, obtain the sample
                # to session map for this clip. The latter will help to restore the output segment
                # during network training and testing. Only the restored output can be used to
                # visualize the output-to-continuous-label line graph.
                intermediate_dict_for_a_clip = {}
                for key, value in compacted_intermediate_dict.items():
                    intermediate_dict_for_a_clip[key] = value[slice(start, end)]
                    if key == 'frame':
                        sample_to_session_map_for_this_clip, relative_session_id, sample_id = \
                            self.get_sample_to_session_map_for_a_clip(
                                value[slice(start, end)], sample_id, relative_session_id,
                                session_having_feeltrace, session_id_of_a_subject)

                clip_info_dict[clip_id] = sample_to_session_map_for_this_clip
                clip_list.append(intermediate_dict_for_a_clip)

            compacted_dict[index] = clip_list
            subject_clip_sample_info[subject_id] = clip_info_dict

        return compacted_dict, sliced_length_dict, subject_clip_sample_info

    def get_sample_to_session_map_for_a_clip(
            self, frame_list, sample_id, relative_session_id,
            session_having_feeltrace, session_id_of_this_subject):

        old_session_id = max(relative_session_id, 0)

        session_id_list = []
        sample_id_list = []

        for count, frame_file in enumerate(frame_list):
            if count % self.time_depth == 0:
                relative_session_id = np.where(
                    self.dataset_info['trial_id'][session_having_feeltrace][session_id_of_this_subject]
                    == self.get_session(frame_file))[0][0]

                if relative_session_id != old_session_id:
                    sample_id = 0
                    old_session_id = relative_session_id
                else:
                    sample_id += 1

                session_id_list.append(relative_session_id)
                sample_id_list.append(sample_id)

        dict_for_this_clip = {'session_id': np.asarray(session_id_list), 'sample_id': np.asarray(sample_id_list)}
        return dict_for_this_clip, relative_session_id, sample_id

    @staticmethod
    def get_session(string):
        return int(string.split("-T")[1].split("_align")[0])

    def shuffle_data_filename_and_label_array(self, generated_data_and_file_list):
        r"""
        Shuffle the data clips, and then combine small list into a big one containing
            self.frame_number_to_compact lines.
        :param generated_data_and_file_list: the list to the shuffle and combine.
        :return: the new list that is shuffled and combined.
        """
        # Initialize the dictionary to record the shuffled data.
        data_label_pool = {key: [] for key in generated_data_and_file_list[0]}

        # Iterate each batch of the data list
        for generated_data_and_file in generated_data_and_file_list:
            length = len(generated_data_and_file['data'])

            # Discard very small batch which has less then self.time_depth lines. Such
            # batches are usually the residual of big sessions.
            if length >= self.time_depth:

                # Calculate the window number given the sampling size and strides.
                window_count = (length - self.time_depth) // self.step_size + 1

                # The sampling indices are save as a list of slices. The slices are used to ensure
                # the temporal continuity for every self.time_depth lines.
                slice_list = [slice(start * self.step_size, start * self.step_size + self.time_depth) for start in
                              range(window_count)]

                # Initialize the dictionary to record the sliced list. If self.step_size is smaller than
                # self.time_depth, then the sliced dictionary will be larger than its precedent.
                generated_data_and_file_sliced = {key: [] for key in generated_data_and_file}

                # Combine lines of each slice to the sliced dictionary.
                for slicing in slice_list:
                    intermediate_dict = {key: value[slicing] for key, value in generated_data_and_file.items()}
                    generated_data_and_file_sliced = dict_combine(generated_data_and_file_sliced, intermediate_dict)

                # Stack the elementary lists into a single one for later convenience.
                generated_data_and_file_sliced = {
                    key: (np.hstack(value).tolist() if key == 'data' else np.vstack(value)) for key, value in
                    generated_data_and_file_sliced.items()}

                # Combine the lines for a session into the pool,
                # so that each element in the pool is the data of a session.
                data_label_pool = dict_combine(data_label_pool, generated_data_and_file_sliced)

        # Flatten the elements in the pool, so that all the lines are in a single list. The list will then
        # be shuffled by a indices list in the equal size.
        data_label_pool = {key: (np.hstack(value) if key == 'data' else np.vstack(value)) for key, value in
                           data_label_pool.items()}

        # Count the clips. It will always be a multiplier of self.time_depth because we have done the
        # slicing before.
        sample_count = len(data_label_pool['data']) // self.time_depth

        # Generate a random non-repetitive indices for the clips.
        random_sample_indices = random.sample(range(0, sample_count), sample_count)

        # Shuffle the indices. The indices correspond to the flattened dictionary.
        # The indices are consecutive for each self.time_depth unit.
        shuffled_indices = np.hstack([np.arange(start * self.time_depth, (start + 1) * self.time_depth) for start in
                                      random_sample_indices])

        # Shuffle the flattened dictionary using the shuffled indices.
        data_label_pool = {
            key: list(itemgetter(*shuffled_indices)(value)) if key == 'data' else value[shuffled_indices, :]
            for key, value in data_label_pool.items()}

        # After the shuffling, we have to batch up the long dictionary,
        # so that the loading later will not overflow the memory.
        # Get the length of long dictionary and calculate the batch number.
        length_before_batched = len(data_label_pool['data'])
        batch_number = length_before_batched // self.frame_number_to_compact + 1

        # Initialize the list to store the batched dictionary.
        data_label_pool_batched = []

        # Iterate over the batch.
        for batch in range(batch_number):
            # Determine the batch window.
            start = batch * self.frame_number_to_compact
            end = (batch + 1) * self.frame_number_to_compact

            # Choose lines within a batch window and append to the final list.
            data_label_pool_batched.append({key: value[slice(start, end)] for key, value in data_label_pool.items()})

        return data_label_pool_batched


class DataArrangerAVEC19(DataArrangerSemaine):
    def __init__(self, config, session_id_of_n_fold):
        super().__init__(session_id_of_n_fold, config)

    def get_session_string_list(self):
        r"""
        Generate the filename of a specific session.
        :return: (list), the filename list.
        """
        session_string_list = []

        sessions_having_feeltrace = self.get_session_having_feeltrace()
        for session in sessions_having_feeltrace:
            subject_id = self.dataset_info['subject_id'][session]
            trial_id = self.dataset_info['trial_id'][session]
            session_string = "P{}-T{}".format(str(subject_id).zfill((2)), str(trial_id).zfill((2)))
            session_string_list.append(session_string)

        return session_string_list

    def get_success_indices(self, session_list):
        r"""
        Get the indices of the frames with the face detected.
        :param session_list: (list), the sessions to load.
        :return: (ndarray), the success indices of the selected frames.
        """
        success_indices_list = []
        for session in session_list:
            success_indices_csv_filename = self.get_file_list("_success.csv")[session]
            success_indices_csv_directory = os.path.join(self.root_directory, self.data_folder,
                                                         success_indices_csv_filename)
            success_indices_list.append(pd.read_csv(success_indices_csv_directory).to_numpy()[:, 0])
        return success_indices_list

    @staticmethod
    def adjust_indices_list(downsampling_indices_list, success_indices_list, ndarray_list):
        for index, (downsampling_indices, success_indices, ndarray) in enumerate(zip(downsampling_indices_list, success_indices_list, ndarray_list)):
            print(index)
            valid_indices = np.where(downsampling_indices <= len(success_indices) - 1)[0]
            downsampling_indices_list[index] = downsampling_indices[valid_indices]
            ndarray_list[index] = ndarray[valid_indices]

        return downsampling_indices_list, ndarray_list

    def get_ndarraies(self, session_list):
        r"""
        Get the continuous label corresponding to the selected frames.
        :param session_list: (list), the sessions to load.
        :return: (ndarray), the continuous label of the selected frames.
        """
        continuous_label_list = []
        for session in session_list:
            continuous_label_csv_filename = self.get_file_list("_continuous_label.csv")[session]
            continuous_label_csv_directory = os.path.join(self.root_directory, self.data_folder,
                                                          continuous_label_csv_filename)
            continuous_label_list.append(pd.read_csv(continuous_label_csv_directory, sep=";", usecols=['arousal','valence']).to_numpy())
        return continuous_label_list

    def generate_data_filename_and_label_array(
            self, session_id_of_subjects, downsampling_interval=1):
        r"""
        Finally, generate a dictionary to save the successful frame filenames, and the corresponding
            ndarraies.
        :param session_id_of_subjects: (list), the session to load.
        :param downsampling_interval: (int), for every downsampling_interval steps will the sample perform.
        :return: (dict), the dictionary containing the data filenames and their continuous labels.
        """
        session_having_feeltrace = self.get_session_having_feeltrace()
        subject_id_for_sessions_having_feeltrace = self.dataset_info['subject_id'][session_having_feeltrace]
        session_list = np.hstack(session_id_of_subjects)

        # Get the frame filenames, their associated ndarraies and success indices.
        # Note that extra feature in addition to facial frames can be stored in the ndarraies,
        # i.e., the continuous_label_list currently. In this case, the argument continuous_label_list should
        # include the extra features (e.g., the action unit) from earlier processing (i.e., self.get_continuous_label())
        frame_jpg_list = self.get_frame_jpg_list(session_list)
        ndarray_list = self.get_ndarraies(session_list)
        success_indices_list = self.get_success_indices(session_list)

        # Obtain the length list for each video (consider only the frames with faces detected).
        length_list = self.dataset_info['frame_count']

        # Generate the pointers for downsampling.
        downsampling_indices_list = self.generate_downsampling_indices(
            length_list, downsampling_interval)

        downsampling_indices_list, ndarray_list = self.adjust_indices_list(
            downsampling_indices_list, success_indices_list, ndarray_list)

        # Apply the downsampling indices on the success indices.
        success_indices_list = [success_indices[downsampling_indices]
                                for success_indices, downsampling_indices in
                                zip(success_indices_list, downsampling_indices_list)]

        # Apply the success_indices on their associated frames and ndarraies.
        success_frame_jpg_list = [list(itemgetter(*success_indices)(frame_jpg_list))
                                  for success_indices, frame_jpg_list in
                                  zip(success_indices_list, frame_jpg_list)]
        success_ndarray_list = ndarray_list

        # Initialize an empty list to record.
        generated_list = []

        sliced_length_dict = {str(key): [] for key in subject_id_for_sessions_having_feeltrace}
        # Synchronously process the frame filename list and ndarray list
        for success_frame_jpgs, success_ndarraies, subject_id in zip(
                success_frame_jpg_list, success_ndarray_list, subject_id_for_sessions_having_feeltrace):

            length = len(success_frame_jpgs)
            sliced_length_list = []
            generated_data_and_file = {'frame': success_frame_jpgs, 'ndarray': success_ndarraies}
            # Discard very small batch which has less then self.time_depth lines. Such
            # batches are usually the residual of big sessions.
            if length >= self.time_depth:

                # Calculate the window number given the sampling size and strides.
                window_count = (length - self.time_depth) // self.step_size + 1

                # The sampling indices are save as a list of slices. The slices are used to ensure
                # the temporal continuity for every self.time_depth lines.
                slice_list = [slice(start * self.step_size, start * self.step_size + self.time_depth) for start in
                              range(window_count)]

                # Initialize the dictionary to record the sliced list. If self.step_size is smaller than
                # self.time_depth, then the sliced dictionary will be larger than its precedent.
                generated_data_and_file_sliced = {key: [] for key in ["frame", "ndarray"]}

                # Combine lines of each slice to the sliced dictionary.
                for slicing in slice_list:
                    intermediate_dict = {key: value[slicing] for key, value in generated_data_and_file.items()}
                    generated_data_and_file_sliced = dict_combine(generated_data_and_file_sliced, intermediate_dict)

                # Stack the elementary lists into a single one for later convenience.
                generated_data_and_file_sliced = {
                    key: (np.hstack(value).tolist() if key == 'frame' else np.vstack(value)) for key, value in
                    generated_data_and_file_sliced.items()}

                generated_list.append(generated_data_and_file_sliced)
                sliced_length_dict[str(subject_id)].append((window_count - 1) * self.step_size + self.time_depth)

        compacted_dict = {}

        subject_id_list = self.get_subject_id_having_feeltrace()
        subject_clip_sample_info = {}

        for index, session_id_of_a_subject in enumerate(session_id_of_subjects):

            relative_session_id = -1
            subject_id = subject_id_list[index]

            # Process the things for one subject at a time.
            data_and_file_for_one_subject = generated_list[index]

            # compacted_intermediate_dict = {key: [] for key in ["frame", "ndarray"]}
            # for data_and_file_for_one_session in data_and_file_for_one_subject:
            compacted_intermediate_dict = data_and_file_for_one_subject

            # Stack the elementary lists into a single one for later convenience.
            compacted_intermediate_dict = {
                key: (np.hstack(value).tolist() if key == 'frame' else np.vstack(value)) for key, value in
                compacted_intermediate_dict.items()}

            # This variable (sample_id) is crucial. It records the indices relative to each session of a subject.
            # A window contain time_depth frames, they constitute a sample.
            # Later, when we obtain the output from the network, we have to place that output
            # segment to the corresponding position of a session-wise long array.
            # This variable can help to determine the correct positions for each output segment.
            # It records the indices relative to each session of a subject. So that even if a clip can occasionally
            # have data from two sessions (this happens when a clip contains more than one samples),
            # we can still retrieve the sample-wise affiliation according to this variable.
            sample_id = -1

            break_flag = False
            clip_list = []

            # Split the data of a given session to several batches, each with a smaller size so that
            # they can be loaded by a commercial PC or server flexibly.

            clip_info_dict = {}
            for clip_id in range(len(compacted_intermediate_dict['frame']) // self.frame_number_to_compact + 1):

                if break_flag:
                    break

                # Determine the batch window. Note that the window amount must be
                # the multiplier of self.time_depth
                start = clip_id * self.frame_number_to_compact
                end = (clip_id + 1) * self.frame_number_to_compact
                if len(compacted_intermediate_dict['frame']) - end <= self.frame_number_to_compact * 0.2:
                    end = len(compacted_intermediate_dict['frame'])
                    break_flag = True

                # Get the frame and continuous label for a clip. Meanwhile, obtain the sample
                # to session map for this clip. The latter will help to restore the output segment
                # during network training and testing. Only the restored output can be used to
                # visualize the output-to-continuous-label line graph.
                intermediate_dict_for_a_clip = {}
                for key, value in compacted_intermediate_dict.items():
                    intermediate_dict_for_a_clip[key] = value[slice(start, end)]
                    if key == 'frame':
                        sample_to_session_map_for_this_clip, relative_session_id, sample_id = \
                            self.get_sample_to_session_map_for_a_clip(
                                value[slice(start, end)], sample_id, relative_session_id,
                                session_having_feeltrace, session_id_of_a_subject)

                clip_info_dict[clip_id] = sample_to_session_map_for_this_clip
                clip_list.append(intermediate_dict_for_a_clip)

            compacted_dict[index] = clip_list
            subject_clip_sample_info[subject_id] = clip_info_dict

        return compacted_dict, sliced_length_dict, subject_clip_sample_info

class EmotionalVideoDataset(Dataset):
    r"""
    A generic Dataset class for videos. It is designed for processing videos of one subject.
    It is designed for subject-wise scenario.
    It basically read n videos of one subject. For one video, there are multiple clips, each with m frames.
        This class concatenates all the nxm clips straightforwardly, and then load them one at a time.
    According to the practice, it may take around 3 GB even loading only one video whose length
        is about 1 min. Therefore, the 'batch_size' of this class is set to 1 for it to be functional in a commercial
        desktop with 16 GB memory.
    This class is used to load video clips, so that the deep features for each video clip can be extracted. It is slow
            --because the clips are in video format, not images, and space-consuming --because the ratio of frames to
            labels are time_depth, which is 16 by default.
    """

    def __init__(self, subject, batch_index, config, transform=None):
        self.root_directory = config['root_directory']
        self.dataset_directory = config['dataset_directory']
        self.dataset_info_extension = config['dataset_info_extension']
        self.dataset_info = load_single_pkl(self.root_directory, '*')
        self.annotation_extension = config['annotation_extension']
        self.video_extension = config['video_extension']
        self.deep_feature_extension = config['deep_feature_extension']
        self.channels = config['channels']
        self.time_depth = config['time_depth']
        self.input_width = config['input_width']
        self.input_height = config['input_height']
        self.fps = config['fps']
        self.overlap = config['overlap']
        self.time_window = config['time_window']
        self.output_width = config['output_width']
        self.output_height = config['output_height']
        self.mean = config['mean']
        self.std = config['std']
        self.batch_size = config['batch_size']
        self.label_dimension = config['label_dimension']
        self.label_column = config['label_column']
        self.label_skip_first_n_rows = config['label_skip_first_n_rows']
        self.subject = subject
        self.batch_index = batch_index
        self.transform = transform

        self.batch_lengths, self.max_length = self.get_length_of_a_batch()
        self.label_index = self.get_label_dimension_index()
        self.init_function()

    def init_function(self):
        self.videos = self.read_video()
        self.labels = self.read_label()

    def get_label_dimension_index(self):
        r"""
        Find the index of emotion dimension. For example, if the label file is formatted as
            2 column --Valence and Arousal, and we would like to deal with Valence, then this
            function should return 0.
        :return: (int), the index of the emotion dimension stored in the label file.
        """
        label_index = np.int32([self.label_column[emotion] for emotion in self.label_dimension])
        return label_index

    def get_length_of_a_batch(self):
        r"""
        Count the label length of all the trials for the current subject.
            It also return the max length.
        :return: (list) and (int), the length for each trial of the subject and the maximum.
        """
        base_index = np.where(self.dataset_info['subject'] == self.subject)[0][0]
        start_index = base_index + self.batch_size * self.batch_index
        end_index = base_index + self.batch_size * (self.batch_index + 1)
        batch_lengths = self.dataset_info['continuous_label_length'][start_index:end_index]
        max_length = max(batch_lengths)
        return batch_lengths, max_length

    def get_files(self, extension, suffix=""):
        r"""
        Get all the files of the subject by the extension and suffix .
        :param extension:  (str), the extension of a file.
        :param suffix:  (str), the suffix of a file.
        :return:
        """
        folder_name = self.dataset_directory

        start_index = self.batch_size * self.batch_index
        end_index = self.batch_size * (self.batch_index + 1)
        trial_list_of_this_subject = np.where(self.dataset_info['subject'] == self.subject)

        trials = self.dataset_info['trial'][trial_list_of_this_subject]

        file_list = [os.path.join(self.root_directory,
                                  self.dataset_directory, str(self.subject),
                                  str(trial) + suffix + extension)
                     for trial in trials][start_index:end_index]

        return file_list

    def read_label(self):
        r"""
        Load the labels for a given list of label files.
        :return: (float ndarray), the labels as an ndarray.
        """
        label_files = self.get_files(self.annotation_extension)
        labels = torch.FloatTensor(
            len(self.label_index), sum(self.batch_lengths)
        )

        start_frame_index = 0
        for file_index in range(len(label_files)):
            label = torch.FloatTensor(
                np.loadtxt(label_files[file_index], delimiter=' ', unpack=True,
                           skiprows=self.label_skip_first_n_rows)[self.label_index, :])

            labels[:, start_frame_index:start_frame_index + label.shape[1]] = label
            start_frame_index += self.batch_lengths[file_index]
        return labels

    def read_video(self):
        r"""
        Load the videos for a given list of video files.
        :return:  (uint8 ndarray), the videos as an ndarray.
        """
        video_files = self.get_files(self.video_extension)
        frames = torch.FloatTensor(
            self.channels, sum(self.batch_lengths) * self.time_depth, self.output_width, self.output_height
        )

        start_frame_index = 0
        for file_index in range(len(video_files)):
            cap = cv2.VideoCapture(video_files[file_index])

            # start_frame_index = file_index * self.max_length
            frame_number = self.batch_lengths[file_index] * self.time_depth
            print(frame_number)
            for frame_index in range(frame_number):
                ret, frame = cap.read()
                if ret:
                    # frame = frame.astype(np.float32) / 255.

                    if self.transform:
                        frame = self.transform(frame)

                    frames[:, start_frame_index + frame_index, :, :] = frame
                else:
                    raise SystemExit(
                        "Failed to load the {}-th frame of {}".format(frame_index, video_files[file_index]))
            cap.release()
            start_frame_index += frame_number
        return frames

    def __getitem__(self, idx):
        r"""
        Get a video clips at a time.
        :param idx: the index of the item to be loaded.
        :return:  a sample dictionary.
        """
        sample = {
            "clip": self.videos[:, idx * self.time_depth: (idx + 1) * self.time_depth, :, :],
            "label": self.labels[:, idx]
        }
        return sample

    def __len__(self):
        r"""
        The length of the sample.
        :return: the length of the label.
        """
        return self.labels.shape[1]


class EmotionalSequentialDataset(Dataset):
    r"""
    A generic class to load features and labels. It loads all the deep features and the labels of n trials.
    It is designed for subject-wise scenario.
    Since the deep features are much smaller than the videos, it can load everything in one go.
    """

    def __init__(self, subjects, config, transform=None):
        self.root_directory = config['root_directory']
        self.dataset_folder = config['dataset_folder']
        self.subjects = subjects
        self.deep_feature_suffix = config['deep_feature_suffix']
        self.deep_feature_extension = config['deep_feature_extension']
        self.dataset_info_filename = config['dataset_info_filename']
        self.annotation_extension = config['annotation_extension']
        self.batch_size = config['batch_size']
        self.label_column = config['label_column']
        self.label_dimension = config['label_dimension']
        self.label_skip_first_n_rows = config['label_skip_first_n_rows']
        self.dataset_info = load_pkl_file(os.path.join(self.root_directory, self.dataset_info_filename))
        self.label_index = self.get_label_dimension_index()
        self.init_function()

    def init_function(self):
        self.features = self.read_deep_feature()
        self.labels = self.read_label()

    def get_label_dimension_index(self):
        label_index = np.int32([self.label_column[emotion] for emotion in self.label_dimension])
        return label_index

    def get_indices_by_subject_id(self):
        r"""
        Get the corresponding indices for given subjects.
        :return: (list), the indices of the data to load, given the subject list.
        """
        indices = np.concatenate([np.where(self.dataset_info['subject'] == subject)
                                  for subject in self.subjects], axis=-1)[0]
        return indices

    def get_files(self, extension, suffix=''):
        indices = self.get_indices_by_subject_id()
        file_list = [
            os.path.join(self.root_directory, self.dataset_folder, str(subject), str(trial) + suffix + extension)
            for subject, trial in zip(self.dataset_info['subject'][indices], self.dataset_info['trial'][indices])]
        return file_list

    def read_deep_feature(self):
        r"""
        Load the deep feature in one go.
        :return: (list of ndarray), all the deep features given the file list. Note, they are variable length, and will then
            be padded by the dataloader.
        """
        file_list = self.get_files(self.deep_feature_extension, self.deep_feature_suffix)
        feature_list = []

        for feature_file in file_list:
            if os.path.isfile(feature_file):
                # Read the features of a trial
                h5 = h5py.File(feature_file, 'r')
                feature = h5['feature'][()]
                h5.close()
                feature_list.append(feature)
            else:
                print("{} is missing.".format(feature_file))
        return feature_list

    def read_label(self):
        r"""
        The same with the last function, but for labels.
        :return:
        """
        file_list = self.get_files(self.annotation_extension)

        label_list = []

        for label_file in file_list:
            if os.path.isfile(label_file):
                # Read the label of a trial
                label = np.loadtxt(label_file, delimiter=' ', unpack=True,
                                   skiprows=self.label_skip_first_n_rows)[self.label_index, :].T
                label_list.append(label)

        return label_list

    def __getitem__(self, idx):
        r"""
        Return the features and labels for one trial as a sample.
        :param idx: the index of the sample.
        :return: the samples of a trial.
        """
        return self.features[idx], self.labels[idx]

    def __len__(self):
        r"""
        Return the length of the trials to process.
        :return: how many trials are to be processed.
        """
        return len(self.features)
