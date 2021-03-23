import os
import shutil
import glob
import pickle
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def init_weighted_sampler_and_weights(dataset):
    class_sample_count = np.unique(dataset.targets, return_counts=True)[1]
    weight = 1. / class_sample_count

    samples_weight = weight[dataset.targets]
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(weights=samples_weight.type('torch.DoubleTensor'), num_samples=len(samples_weight))
    return sampler, samples_weight


def dict_combine(main_dict, new_dict):
    r"""
    Combine two dictionaries having the same keys.
    :param main_dict: (dict), the main dictionary to be appended.
    :param new_dict: (dict), the given dictionary to append.
    :return: (dict), the combined dictionary.
    """
    for (key, value) in main_dict.items():
        if new_dict[key] != [] or "":
            main_dict[key].append(new_dict[key])

    return main_dict

def get_video_length(video_filename):
    video = cv2.VideoCapture(video_filename)
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return count


def copy_file(input_filename, output_filename):
    if not os.path.isfile(output_filename):
        shutil.copy(input_filename, output_filename)


def save_pkl_file(directory, filename, data):
    os.makedirs(directory, exist_ok=True)
    fullname = os.path.join(directory, filename)

    with open(fullname, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_single_pkl(directory, filename=None, extension='.pkl'):
    r"""
    Load one pkl file according to the filename.
    """
    if filename is not None:
        fullname = os.path.join(directory, filename + extension)
    else:
        fullname = directory if directory.endswith(".pkl") else directory + ".pkl"

    fullname = glob.glob(fullname)[0]

    with open(fullname, 'rb') as f:
        pkl_file = pickle.load(f)

    return pkl_file


def load_single_csv(directory, filename=None, extension='.csv', header=0):
    if filename is not None:
        fullname = os.path.join(directory, filename + extension)
    else:
        fullname = directory

    fullname = glob.glob(fullname)[0]
    csv_data = pd.read_csv(fullname, header=header)
    return csv_data


def get_filename_from_full_path(full_path):
    return full_path.split(os.sep)[-1]


def get_filename_from_a_folder_given_extension(folder, extension):
    file_list = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            file_list.append(os.path.join(folder, file))

    return file_list


def detect_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda:1')
    return device


def select_gpu(index):
    r"""
    Choose which gpu to use.
    :param index: (int), the index corresponding to the desired gpu. For example,
        0 means the 1st gpu.
    """
    torch.cuda.set_device(index)


def set_cpu_thread(number):
    r"""
    Set the maximum thread of cpu for torch module.
    :param number: (int), the number of thread allowed, usually 1 is enough.
    """
    torch.set_num_threads(number)
