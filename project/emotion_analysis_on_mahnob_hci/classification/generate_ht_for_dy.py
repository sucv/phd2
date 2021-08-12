import numpy as np
import h5py

import pickle
import os



def frame(data, window_length, hop_length):
    """Convert array into a sequence of successive possibly overlapping frames.
  An n-dimensional array of shape (num_samples, ...) is converted into an
  (n+1)-D array of shape (num_frames, window_length, ...), where each frame
  starts hop_length points after the preceding one.
  This is accomplished using stride_tricks, so the original data is not
  copied.  However, there is no zero-padding, so any incomplete frames at the
  end are not included.
  Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.
  Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.
  """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

load_path = r"E:\Mahnob_full\dy_h5"
save_path = r"E:\full"
current_subject = 1
class_label_path = r"E:\Mahnob_full\class_label_dy.pkl"

with open(class_label_path, 'rb') as f:
    labels = pickle.load(f)

data_list = []
arousal_list = []
valence_list = []
for index, folder in enumerate(os.listdir(load_path)):
    path = os.path.join(load_path, folder)
    subject = int(folder.split("-")[0][1:])

    if subject != current_subject:
        filename = "sub" + str(subject) + ".pkl"
        file_path = os.path.join(save_path, filename)

        data_of_this_subject = {
            'data': data_list,
            'arousal': arousal_list,
            'valence': valence_list
        }

        if not os.path.isfile(file_path):
            with open(file_path, 'wb') as handle:
                pickle.dump(data_of_this_subject, handle)

        current_subject = subject

        data_list = []
        arousal_list = []
        valence_list = []

    eeg_file = os.path.join(path,"eeg_raw.npy")
    label_file = os.path.join(path, "class_label.npy")
    eeg_data = np.load(eeg_file)
    data = frame(eeg_data, 1024, 64)[:-12, :, :]
    data = np.transpose(data, (0, 2, 1))

    length = len(data)
    arousal = np.tile(labels[folder]['Arousal'], length)
    valence = np.tile(labels[folder]['Valence'], length)


    data_list.append(data)
    arousal_list.append(arousal)
    valence_list.append(valence)

    if index == len(os.listdir(load_path)) - 1:


        filename = "sub" + str(subject) + ".pkl"
        file_path = os.path.join(save_path, filename)
        if not os.path.isfile(file_path):
            data_of_this_subject = {
                'data': data_list,
                'arousal': arousal_list,
                'valence': valence_list
            }


            with open(file_path, 'wb') as handle:
                pickle.dump(data_of_this_subject, handle)

    print(index)