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


number_to_emotion_tag_dict = {
    "0": "Neutral",
    "1": "Anger",
    "2": "Disgust",
    "3": "Fear",
    "4": "Joy",
    "5": "Sadness",
    "6": "Surprise",
    "11": "Amusement",
    "12": "Anxiety"
}

emotion_tag_to_valence_class = {
    "Neutral": "Neutral valence",
    "Anger": "Unpleasant",
    "Disgust": "Unpleasant",
    "Fear": "Unpleasant",
    "Joy": "Pleasant",
    "Sadness": "Unpleasant",
    "Surprise": "Neutral valence",
    "Amusement": "Pleasant",
    "Anxiety": "Unpleasant"
}

emotion_tag_to_arousal_class = {
    "Neutral": "Calm",
    "Anger": "Activated",
    "Disgust": "Calm",
    "Fear": "Activated",
    "Joy": "Medium arousal",
    "Sadness": "Calm",
    "Surprise": "Activated",
    "Amusement": "Medium arousal",
    "Anxiety": "Activated"
}

valence_class_to_number = {
    "Unpleasant": 0,
    "Neutral valence": 1,
    "Pleasant": 2
}

arousal_class_to_number = {
    "Calm": 0,
    "Medium arousal": 1,
    "Activated": 2
}
