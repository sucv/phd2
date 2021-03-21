from base.utils import load_single_pkl
import os


class VideoEmoRegressionArranger(object):
    def __init__(self, config):
        self.root_directory = config['remote_root_directory']
        self.npy_folder = config['npy_folder']
        self.window_length = config['window_length']
        self.hop_size = config['hop_size']
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
