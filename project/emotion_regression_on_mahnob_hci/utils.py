import numpy as np
import pandas as pd


def read_start_end_from_mahnob_tsv(tsv_file_list):
    r"""
    Get the start and end indices of the stimulated video frames by
        reading the tsv file from MAHNOB-HCI dataset.
    :param tsv_file_list: (list), the tsv file list.
    :return: (ndarray), the start and end indices of the stimulated frames,
        for later video trimming.
    """
    start_end_array = np.zeros((len(tsv_file_list), 1, 2), dtype=int)
    for index, tsv_file in enumerate(tsv_file_list):
        data = pd.read_csv(tsv_file, sep='\t', skiprows=23)
        end = data[data['Event'] == 'MovieEnd'].index[0]
        start_end_array[index, :, 1] = end
    return start_end_array