import os
import pickle


class GenericImagePreprcessing(object):

    def image_preprocessing(self):
        pass


class GenericImagePreprcessingForNFoldCV(GenericImagePreprcessing):

    def get_image_list(self):
        pass

    @staticmethod
    def get_expression_category(filename):
        pass

    @staticmethod
    def get_subject_id(filename):
        pass

    def copy_paste(self, filename):
        pass


class GenericVideoPreprocessing(object):
    def __init__(self, opts):
        self.config = opts
        self.root_directory = opts['local_root_directory']
        self.raw_data_folder = opts['raw_data_folder']
        self.openface_output_folder = opts['openface_output_folder']
        self.npy_folder = opts['npy_folder']
        self.emotion_dimension = opts['emotion_dimension']
        self.downsampling_interval_dict = opts['downsampling_interval_dict']
        self.target_fps = opts['target_fps']
        self.window_length = opts['window_length']
        self.hop_size = opts['hop_size']
        self.frame_size = opts['frame_size']
        self.openface_config = opts['openface_config']
        self.dataset_info = self.init_dataset_info()

    def count_session(self):
        pass

    def get_continuous_label_indicator(self):
        pass

    def get_eeg_indicator(self):
        pass

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

    def video_preprocessing(self):
        pass

    def feature_preprocessing(self):
        pass

    def eeg_preprocessing(self):
        pass

    def label_preprocessing(self):
        pass

    def create_npy_for_continuous_label(self):
        pass

    def create_npy_for_feature(self):
        pass

    def create_npy_for_eeg(self):
        pass

    def create_npy_for_frame(self):
        pass

    @staticmethod
    def init_dataset_info():
        dataset_info = {
            "subject_id": [],
            "trial_id": [],
        }
        return dataset_info





