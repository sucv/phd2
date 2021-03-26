from base.preprocessing import GenericImagePreprcessing, GenericImagePreprcessingForNFoldCV
from base.facial_landmark import facial_image_crop_by_landmark
from base.utils import load_single_csv, copy_file
from base.video import OpenFaceController

import argparse
import os
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image


class PreprocessingAffectNet(GenericImagePreprcessing):
    def __init__(self, config):
        super().__init__()
        self.root_directory = config['local_root_directory']
        self.image_folder = config['local_image_folder']
        self.output_directory = config['local_output_directory']
        self.label_csv_list = ["validation", "training"]
        self.emotion_dict = {0: "Neutral", 1: "Happy", 2: "Sad",
                             3: "Surprise", 4: "Fear", 5: "Disgust",
                             6: "Anger", 7: "Contempt", 8: "None",
                             9: "Uncertain", 10: "Non-Face"}

        self.selected_emotion = {"Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"}
        self.config = config
        self.image_preprocessing()

    def image_preprocessing(self):

        landmark_handler = facial_image_crop_by_landmark(self.config)

        for label_csv_file in self.label_csv_list:

            # Make the folder naming universal.
            partition = "train"
            if label_csv_file == "validation":
                partition = "validate"

            csv_data = load_single_csv(self.root_directory, label_csv_file, ".csv")

            for index, data in csv_data.iterrows():

                image_fullname = os.path.join(
                    self.root_directory, self.image_folder, data[0].split("/")[0], data[0].split("/")[1])

                emotion = self.emotion_dict[data[6]]

                if emotion in self.selected_emotion:
                    output_directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
                    os.makedirs(output_directory, exist_ok=True)

                    output_fullname = os.path.join(output_directory, data[0].split("/")[0] + "_" + data[0].split("/")[1])

                    if not os.path.isfile(output_fullname):
                        landmark = self.restore_landmark_to_ndarray(data[5])

                        img_ndarray = np.array(Image.open(image_fullname))
                        croped_image = landmark_handler.crop_image(img_ndarray, landmark)

                        croped_image = Image.fromarray(croped_image)

                        croped_image.save(output_fullname, "JPEG")


    @staticmethod
    def restore_landmark_to_ndarray(landmark_string):
        landmark = np.fromstring(landmark_string, sep=";").astype(np.float).reshape((68, 2))
        return landmark


class PreprocessingRAFDB(GenericImagePreprcessing):
    def __init__(self, config):
        super().__init__()
        self.root_directory = config['local_root_directory']
        self.image_folder = config['local_image_folder']
        self.output_directory = config['local_output_directory']
        self.emotion_dict = {1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Sadness", 6: "Surprise", 7: "Neutral"}
        self.image_preprocessing()

    @staticmethod
    def load_txt(directory, filename, extension=".txt"):

        fullname = os.path.join(directory, filename + extension)
        data = pd.read_csv(fullname, sep=" ", header=None)
        return data

    def load_label(self):
        label = self.load_txt(self.root_directory, "list_patition_label")
        return label

    def image_preprocessing(self):
        label = self.load_label()
        directory = os.path.join(self.root_directory, self.image_folder)

        for index, data in tqdm(label.iterrows()):
            image_fullname = os.path.join(directory, data[0].split(".jpg")[0] + "_aligned" + ".jpg")
            partition = data[0].split("_")[0]
            emotion = self.emotion_dict[data[1]]
            output_directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
            os.makedirs(output_directory, exist_ok=True)
            output_fullname = os.path.join(output_directory, data[0].split(".jpg")[0] + "_aligned" + ".jpg")
            if not os.path.isfile(output_fullname):
                copy_file(image_fullname, output_fullname)


class PreprocessingFER2013(GenericImagePreprcessing):
    def __init__(self, config):
        super().__init__()
        self.root_directory = config['local_root_directory']
        self.root_csv_filename = config['root_csv_filename']
        self.output_directory = config['local_output_directory']
        self.emotion_dict = self.init_emotion_dict()
        self.partition_dict = {"Training": "train", "PrivateTest": "validate", "PublicTest": "test"}
        self.image_preprocessing()

    def init_emotion_dict(self):
        emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
        return emotion_dict

    def load_csv(self, extension):
        r""""
        FER2013 is stored in a csv file.
        """
        csv_data = load_single_csv(
            directory=self.root_directory, filename=self.root_csv_filename, extension=extension)

        return csv_data

    @staticmethod
    def restore_img_from_csv_row(pixel_array):
        img = np.fromstring(pixel_array, dtype=int, sep=" ").reshape((48, 48)).astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img

    def image_preprocessing(self):
        csv_data = self.load_csv(".csv")

        for index, row in tqdm(csv_data.iterrows()):
            emotion = self.emotion_dict[row['emotion']]
            pixels = row['pixels']
            partition = self.partition_dict[row['Usage']]

            directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
            os.makedirs(directory, exist_ok=True)

            fullname = os.path.join(directory, str(index) + ".jpg")
            if not os.path.isfile(fullname):
                img = self.restore_img_from_csv_row(pixels)
                img.save(fullname, "JPEG")


class PreprocessingFerPlus(PreprocessingFER2013):
    def __init__(self, config):
        super().__init__(config)

    def init_emotion_dict(self):

        emotion_dict = {0: "Neutral", 1: "Happiness", 2: "Surprise", 3: "Sadness", 4: "Anger", 5: "Disgust", 6: "Fear",
                        7: "Contempt"}
        return emotion_dict

    def image_preprocessing(self):
        fer_csv_data = self.load_csv(".csv")
        fer_plus_csv_data = self.load_csv("new.csv")

        for (index, fer_row), (_, fer_plus_row) in tqdm(zip(fer_csv_data.iterrows(), fer_plus_csv_data.iterrows())):

            emotion_pointer = np.argmax(fer_plus_row[2:10].array)
            if emotion_pointer < 8:
                labels = np.array(fer_plus_row[2:10].array)
                max_vote = np.max(labels)
                if max_vote > 5:
                # max_vote_emotion = np.where(labels == max_vote)[0]
                # num_max_votes = max_vote_emotion.size
                # if not (num_max_votes >= 3) and not (num_max_votes * max_vote <= 0.5 * num_votes[ii]):

                    emotion = self.emotion_dict[emotion_pointer]
                    pixels = fer_row['pixels']
                    partition = self.partition_dict[fer_row['Usage']]

                    directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
                    os.makedirs(directory, exist_ok=True)

                    fullname = os.path.join(directory, str(index) + ".jpg")
                    if not os.path.isfile(fullname):
                        img = self.restore_img_from_csv_row(pixels)
                        img.save(fullname, "JPEG")


class PreprocessingCKplus(GenericImagePreprcessingForNFoldCV):
    def __init__(self, config):
        super().__init__()
        self.root_directory = config['local_root_directory']
        self.image_folder = config['image_folder']
        self.label_folder = config['label_folder']
        self.openface_output_folder = config['openface_output_folder']
        self.openface_config = config['openface_config']


        self.image_sequence_preprocessing()

    def get_label_list(self):
        label_directory = os.path.join(self.root_directory, self.label_folder)
        label_list = [str(path) for path in Path(label_directory).rglob('*.txt')]
        return label_list

    def image_sequence_preprocessing(self):

        label_list = self.get_label_list()

        for index, label_file in tqdm(enumerate(label_list)):
            subject_id = self.get_subject_id(label_file)
            emotion_category = self.get_expression_category(label_file)

            image_sequence_directory = self.get_image_sequence_directory(label_file)

            output_filename = emotion_category
            output_directory = os.path.join(self.openface_output_folder, subject_id)
            os.makedirs(output_directory, exist_ok=True)

            openface = OpenFaceController(openface_path=self.openface_config['openface_directory'], output_directory=output_directory)
            _ = openface.process_video(
                input_filename=image_sequence_directory, output_filename=output_filename, **self.openface_config)


    def get_image_sequence_directory(self, label_filename):
        label_filename_parts = label_filename.split(os.sep)
        label_filename_parts[2] = self.image_folder
        image_sequence_directory = os.sep.join(label_filename_parts[:-1])
        return image_sequence_directory

    @staticmethod
    def get_subject_id(label_filename):
        subject_id = label_filename.split(os.sep)[3]
        return subject_id

    @staticmethod
    def get_expression_category(label_filename):
        emotion_code = np.loadtxt(label_filename)

        if emotion_code == 0:
            emotion_category = "Neutral"
        elif emotion_code == 1:
            emotion_category = "Anger"
        elif emotion_code == 2:
            emotion_category = "Contempt"
        elif emotion_code == 3:
            emotion_category = "Disgust"
        elif emotion_code == 4:
            emotion_category = "Fear"
        elif emotion_code == 5:
            emotion_category = "Happiness"
        elif emotion_code == 6:
            emotion_category = "Sad"
        elif emotion_code == 7:
            emotion_category = "Surprise"
        else:
            raise ValueError("Unknown expression code!")

        return emotion_category


class PreprocessingOuluCISIA(GenericImagePreprcessingForNFoldCV):
    def __init__(self, config):
        super().__init__()
        self.root_directory = config['local_root_directory']
        self.openface_output_folder = config['openface_output_folder']
        self.openface_config = config['openface_config']
        self.get_video_list()
        self.image_sequence_preprocessing()

    def image_sequence_preprocessing(self):

        video_list = self.get_video_list()

        for index, file in tqdm(enumerate(video_list)):
            if " " in file:
                file = '"' + file + '"'

            subject_id = self.get_subject_id(file)
            emotion_category = self.get_expression_category(file)
            output_filename = emotion_category
            output_directory = os.path.join(self.openface_output_folder, subject_id)
            os.makedirs(output_directory, exist_ok=True)

            openface = OpenFaceController(openface_path=self.openface_config['openface_directory'],
                                          output_directory=output_directory)
            _ = openface.process_video(input_filename=file, output_filename=output_filename, **self.openface_config)



    def get_video_list(self):
        r"""
        Get the videos captured under strong illumination.
        """
        reg_compile = re.compile(r'.+_S_.+')
        video_list = [os.path.join(self.root_directory, file) for file in os.listdir(self.root_directory) if reg_compile.match(file)]
        return video_list

    @staticmethod
    def get_subject_id(video_name):
        subject_id = video_name.split(os.sep)[-1].split("_")[1]
        return subject_id

    @staticmethod
    def get_expression_category(video_name):
        emotion_code = video_name.split(os.sep)[-1].split("_")[-1].split(".avi")[0]

        if emotion_code == "A":
            emotion_category = "Anger"
        elif emotion_code == "D":
            emotion_category = "Disgust"
        elif emotion_code == "F":
            emotion_category = "Fear"
        elif emotion_code == "H":
            emotion_category = "Happiness"
        elif emotion_code == "S1":
            emotion_category = "Surprise"
        elif emotion_code == "S2":
            emotion_category = "Sad"
        else:
            raise ValueError("Unknown expression code!")

        return emotion_category


class PreprocessingRAFD(PreprocessingOuluCISIA):
    def __init__(self, config):
        super().__init__(config)

    def get_video_list(self):
        directory = os.path.join(self.root_directory, "raw_data")
        video_list = [os.path.join(directory, image) for image in os.listdir(directory)]
        return video_list

    @staticmethod
    def get_expression_category(video_name):
        emotion_category = video_name.split(os.sep)[-1].split("_")[4].split(".avi")[0].capitalize()
        return emotion_category

    @staticmethod
    def get_gaze(video_name):
        gaze = video_name.split(os.sep)[-1].split("_")[5].split(".avi")[0].capitalize()[:-4]
        return gaze

    def image_sequence_preprocessing(self):

        video_list = self.get_video_list()

        for index, file in tqdm(enumerate(video_list)):
            if " " in file:
                file = '"' + file + '"'

            subject_id = self.get_subject_id(file)
            emotion_category = self.get_expression_category(file)
            gaze = self.get_gaze(file)
            output_filename = emotion_category + "_" + gaze
            output_directory = os.path.join(self.openface_output_folder, subject_id)
            os.makedirs(output_directory, exist_ok=True)

            openface = OpenFaceController(openface_path=self.openface_config['openface_directory'],
                                          output_directory=output_directory)
            processed_filefullname = openface.process_video(
                input_filename=file, output_filename=output_filename, **self.openface_config)

            self.copy_paste(processed_filefullname)

    def copy_paste(self, filename):
        image_fullname = os.path.join(filename + "_aligned", "frame_det_00_000001.jpg")

        subject_id = filename.split(os.sep)[3]
        emotion_category = filename.split(os.sep)[4].split("_")[0]
        new_filename = filename.split(os.sep)[4].split("_")[1] + ".jpg"
        new_directory = os.path.join(self.root_directory, "Cropped", subject_id, emotion_category)
        os.makedirs(new_directory, exist_ok=True)
        output_image_fullname = os.path.join(new_directory, new_filename)
        copy_file(image_fullname, output_image_fullname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing of Emotion Classification Image Datasets.")
    parser.add_argument("-d", help="Wchich dataset to preprocess? [affectnet, ck+, fer2013, fer+, rafd, rafdb, oulu]", default="rafd")
    args = parser.parse_args()

    if args.d == "affectnet":
        from project.emotion_classification_on_static_image.configs import config_affectnet as config
        pre = PreprocessingAffectNet(config=config)
    elif args.d == "ck+":
        from project.emotion_classification_on_static_image.configs import config_ckplus as config
        pre = PreprocessingCKplus(config=config)
    elif args.d == "fer2013":
        from project.emotion_classification_on_static_image.configs import config_fer2013 as config
        pre = PreprocessingFER2013(config=config)
    elif args.d == "fer+":
        from project.emotion_classification_on_static_image.configs import config_ferplus as config
        pre = PreprocessingFerPlus(config=config)
    elif args.d == "rafd":
        from project.emotion_classification_on_static_image.configs import config_rafd as config
        pre = PreprocessingRAFD(config=config)
    elif args.d == "rafdb":
        from project.emotion_classification_on_static_image.configs import config_rafdb as config
        pre = PreprocessingRAFDB(config=config)
    elif args.d == "oulu":
        from project.emotion_classification_on_static_image.configs import config_oulu as config
        pre = PreprocessingOuluCISIA(config=config)
    else:
        raise ValueError("Unknown dataset!")