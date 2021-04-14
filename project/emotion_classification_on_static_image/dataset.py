from base.dataset import ImageEmoClassificationNFoldArranger

import os
from pathlib import Path
from operator import itemgetter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CKplusArranger(ImageEmoClassificationNFoldArranger):
    def __init__(self, config, num_folds):
        super().__init__(config, num_folds)

    def init_emotion_dict(self):
        return {'Neutral': 0, 'Anger': 1, 'Contempt': 2, 'Disgust': 3,
                'Fear': 4, 'Happiness': 5, 'Sad': 6, 'Surprise': 7}

    def establish_fold(self):
        foldwise_subject_count = self.count_subject_for_each_fold()
        fold_list = [[] for i in range(self.num_folds)]

        start = 0
        for i in range(self.num_folds):
            end = start + foldwise_subject_count[i]
            subject_id_in_this_fold = list(itemgetter(*range(start, end))(self.subject_list))

            for subject_id in subject_id_in_this_fold:
                subject_directory = os.path.join(self.root_directory, subject_id)

                for path in Path(subject_directory).rglob('*.csv'):
                    label = self.emotion_dict[path.name.split(".csv")[0]]
                    image_directory = str(path).split(".csv")[0] + "_aligned"
                    peak_image_directory = [str(image) for image in Path(image_directory).rglob('*.jpg')][-1]
                    fold_list[i].append([label, peak_image_directory])

            start = end

        return fold_list

    def get_subject_list(self):
        subject_list = [folder for folder in os.listdir(self.root_directory)]
        return subject_list

    @staticmethod
    def count_subject_for_each_fold():
        foldwise_subject_count = [12, 12, 12, 12, 12, 12, 12, 12, 12, 10]
        return foldwise_subject_count


class RafdArranger(CKplusArranger):
    def __init__(self, config, num_folds):
        super().__init__(config, num_folds)

    def establish_fold(self):
        foldwise_subject_count = self.count_subject_for_each_fold()
        fold_list = [[] for i in range(self.num_folds)]

        start = 0
        for i in range(self.num_folds):
            end = start + foldwise_subject_count[i]
            subject_id_in_this_fold = list(itemgetter(*range(start, end))(sorted(self.subject_list)))

            for subject_id in subject_id_in_this_fold:
                subject_directory = os.path.join(self.root_directory, subject_id)

                for path in Path(subject_directory).rglob('*.jpg'):
                    label = self.emotion_dict[str(path).split(os.sep)[7]]
                    image_directory = path.parent
                    for image in Path(image_directory).rglob('*.jpg'):
                        fold_list[i].append([label, str(image)])

            start = end

        return fold_list

    @staticmethod
    def count_subject_for_each_fold():
        foldwise_subject_count = [7, 7, 7, 7, 7, 7, 7, 6, 6, 6]
        return foldwise_subject_count

    def init_emotion_dict(self):
        return {'Angry': 0, 'Contemptuous': 1, 'Disgusted': 2, 'Fearful': 3,
                'Happy': 4, 'Neutral': 5, 'Sad': 6, 'Surprised': 7}


class OuluArranger(CKplusArranger):
    def __init__(self, config, num_folds):
        super().__init__(config, num_folds)
        self.emotion_dict = self.init_emotion_dict()

    def init_emotion_dict(self):
        return {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Sad': 4, 'Surprise': 5}

    @staticmethod
    def count_subject_for_each_fold():
        foldwise_subject_count = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        return foldwise_subject_count


class FerplusCrossEntropyClassificationDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.data, self.data_label = self.load_data()

    def load_data(self):

        npy_data = np.load(os.path.join(self.data_path, self.mode + "_data.npy"), mmap_mode='c')
        npy_data_label = np.load(os.path.join(self.data_path, self.mode + "_data_label.npy"), mmap_mode='c')

        return npy_data, npy_data_label

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.Tensor(self.data_label[index])

        return image, label

    def __len__(self):
        return len(self.data)


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