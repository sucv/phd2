import cv2
import subprocess
import os
from base.utils import copy_file
from tqdm import tqdm
import numpy as np


class VideoSplit(object):
    r"""
        A base class to  split video according to a list. For example, given
        [(0, 1000), (1200, 1500), (1800, 1900)] as the indices, the associated
        frames will be split and combined  to form a new video.
    """

    def __init__(self, input_filename, output_filename, trim_range):
        r"""
        The init function of the class.
        :param input_filename: (str), the absolute directory of the input video.
        :param output_filename:  (str), the absolute directory of the output video.
        :param trim_range: (list), the indices of useful frames.
        """

        self.input_filename = input_filename
        self.output_filename = output_filename

        self.video = cv2.VideoCapture(self.input_filename)

        # The frame count.
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # The fps count.
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        # The size of the video.
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # The range to trim the video.
        self.trim_range = trim_range

        # The settings for video writer.
        self.codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(output_filename,
                                      self.codec, self.fps,
                                      (self.width, self.height), isColor=True)

    def jump_to_frame(self, frame_index):
        r"""
        Jump to a specific frame by its index.
        :param frame_index:  (int), the index of the frame to jump to.
        :return: none.
        """
        self.video.set(1, frame_index)

    def read(self, start, end, visualize):
        r"""
        Read then write the frames within (start, end) one frame at a time.
        :param start:  (int), the starting index of the range.
        :param end:  (int), the ending index of the range.
        :param visualize:  (boolean), whether to visualize the process.
        :return:  none.
        """

        # Jump to the starting frame.
        self.jump_to_frame(start)

        # Sequentially write the next end-start frames.
        for index in range(end - start):
            ret, frame = self.video.read()
            self.writer.write(frame)
            if ret and visualize:
                cv2.imshow('frame', frame)
                # Hit 'q' on the keyboard to quit!
                cv2.waitKey(1)

    def combine(self, visualize=False):
        r"""
        Combine the clips  into a single video.
        :param visualize: (boolean), whether to visualize the process.
        :return:  none.
        """

        # Iterate over the pair of start and end.
        for clip_index in range(self.trim_range.shape[0]):
            (start, end) = self.trim_range[clip_index]
            self.read(start, end, visualize)

        self.video.release()
        self.writer.release()
        if visualize:
            cv2.destroyWindow('frame')


def change_video_fps(videos, target_fps):
    r"""
    Alter the frame rate of a given video.
    :param videos:  (list),a list of video files to process.
    :param target_fps:  (float), the desired fps.
    :return: (list), the list of video files after the process.
    """
    output_video_list = []
    print("Changing video fps...")
    # Iterate over the video list.
    for video_name in tqdm(videos):

        # Define the new file name by adding the fps string at the rear before the extension.
        output_video_name = video_name[:-4] + '_fps' + str(target_fps) + video_name[-4:]

        # If the new name already belongs to a file, then do nothing.
        if os.path.isfile(output_video_name):
            # print("Skipped fps conversion for video {}!".format(str(index+1)))
            pass

        # If not, call the ffmpeg tools to change the fps.
        # -qscale:v 0 can preserve the quality of the frame after the recoding.
        else:
            input_codec = " xvid "
            if ".mp4" in video_name:
                input_codec = " mp4v "
            command = "ffmpeg -i {} -filter:v fps=fps={} -c:v mpeg4 -vtag {} -qscale:v 0 {}".format(
                '"' + video_name + '"', str(target_fps), input_codec,
                '"' + output_video_name + '"')
            subprocess.call(command, shell=True)

        output_video_list.append(output_video_name)
    return output_video_list


def combine_annotated_clips(
        videos,
        clip_ranges,
        direct_copy=False,
        visualize=False
):
    output_video_list = []

    print("combining annotated clips...")
    for video_idx, input_video_name in tqdm(enumerate(videos)):

        # Define the new file name by adding the tag at the rear before the extension.
        output_video_name = input_video_name[:-4] + '_combined' + input_video_name[-4:]

        # If the new name already belongs to a file, then do nothing.
        if os.path.isfile(output_video_name):
            print("Skipped video combination for video {}!".format(str(video_idx + 1)))
            pass

        # If not, call the video combiner.
        else:
            if not direct_copy:
                video_split = VideoSplit(input_video_name, output_video_name, clip_ranges[video_idx])
                video_split.combine(visualize)
            else:
                copy_file(input_video_name, output_video_name)

        output_video_list.append(output_video_name)
    return output_video_list


class OpenFaceController(object):
    def __init__(self, openface_path, output_directory):
        self.openface_path = openface_path
        self.output_directory = output_directory

    def get_openface_command(self, **kwargs):
        openface_path = self.openface_path
        input_flag = kwargs['input_flag']
        output_features = kwargs['output_features']
        output_action_unit = kwargs['output_action_unit']
        output_image_flag = kwargs['output_image_flag']
        output_image_size = kwargs['output_image_size']
        output_image_format = kwargs['output_image_format']
        output_filename_flag = kwargs['output_filename_flag']
        output_directory_flag = kwargs['output_directory_flag']
        output_directory = self.output_directory
        output_image_mask_flag = kwargs['output_image_mask_flag']

        command = openface_path + input_flag + " {input_filename} " + output_features \
                  + output_action_unit + output_image_flag + output_image_size \
                  + output_image_format + output_filename_flag + " {output_filename} " \
                  + output_directory_flag + output_directory + output_image_mask_flag
        return command

    def get_indices_having_continuous_label(self, dataset_info):
        # If dataset_info has no key named feeltrace_bool.
        indices_having_continuous_label = range(len(dataset_info['subject_id']))
        # Otherwise, exclude those indices having no continuous label.
        if 'feeltrace_bool' in dataset_info:
            indices_having_continuous_label = np.where(dataset_info['feeltrace_bool'] == 1)[0]

        return indices_having_continuous_label

    @staticmethod
    def get_output_filename(dataset_info, session_id):
        # If the session pattern has no trial information.
        output_filename = "P{}".format(dataset_info['subject_id'][session_id])
        # Otherwise consider the trial information.
        if 'trial_id' in dataset_info:
            output_filename = "P{}-T{}".format(dataset_info['subject_id'][session_id],
                                               dataset_info['trial_id'][session_id])

        return output_filename

    def process_video(self, input_filename, output_filename, **kwargs):

        # Quote the file name if spaces occurred.
        if " " in input_filename:
            input_filename = '"' + input_filename + '"'

        command = self.get_openface_command(**kwargs)
        command = command.format(
            input_filename=input_filename, output_filename=output_filename)

        if not os.path.isfile(os.path.join(self.output_directory, output_filename + ".csv")):
            subprocess.call(command, shell=True)

        return os.path.join(self.output_directory, output_filename)

    def process_video_list(self, video_list, dataset_info, **kwargs):
        processed_video_list = []
        indices_having_continuous_label = self.get_indices_having_continuous_label(dataset_info)

        for i, file in tqdm(enumerate(video_list)):
            session_id = indices_having_continuous_label[i]
            output_filename = self.get_output_filename(dataset_info, session_id)
            processed_filename = self.process_video(input_filename=file, output_filename=output_filename, **kwargs)
            processed_video_list.append(processed_filename)

        return processed_video_list
