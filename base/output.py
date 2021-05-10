import os

import statistics
import numpy as np
import matplotlib.pyplot as plt


class ContinuousOutputHandlerNPYTrial(object):
    def __init__(self, emotion_dimension):
        self.emotion_dimension = emotion_dimension
        self.seen_trials = []
        self.trialwise_records = {}
        self.partition_records = {emotion: [] for emotion in self.emotion_dimension}

    def update_output_for_seen_trials(self, output, trials, indices, lengths):

        for i, trial in enumerate(trials):

            # If this is the first time to record the output for trial
            if trial not in self.seen_trials:
                self.seen_trials.append(trial)
                self.trialwise_records[trial] = self.create_list_for_a_trial(lengths[i])

            index = indices[i]
            for j, emotion in enumerate(self.emotion_dimension):
                [self.trialwise_records[trial][emotion][index[k]].append(output[i, k, j])  for k, data in enumerate(output[i, :, j])]

    def average_trial_wise_records(self):

        for trial in self.seen_trials:
            for emotion in self.emotion_dimension:
                length = len(self.trialwise_records[trial][emotion])

                for i in range(length):
                    self.trialwise_records[trial][emotion][i] = statistics.mean(
                        self.trialwise_records[trial][emotion][i])

                self.trialwise_records[trial][emotion] = np.asarray(self.trialwise_records[trial][emotion])

    def concat_records(self):
        for trial in self.seen_trials:
            for emotion in self.emotion_dimension:
                self.partition_records[emotion].extend(self.trialwise_records[trial][emotion])

        for emotion in self.emotion_dimension:
            self.partition_records[emotion] = np.asarray(self.partition_records[emotion])

    def create_list_for_a_trial(self, length):
        trial_record = {emotion: [[] for i in range(length)] for emotion in self.emotion_dimension}
        return trial_record

class ContinuousOutputHandlerNPY(object):
    def __init__(self, length_to_track, emotion_dimension):
        self.length_to_track = length_to_track
        self.emotion_dimension = emotion_dimension
        self.subjectwise_dict = self.init_subjectwise_dict()
        self.sessionwise_dict = self.init_sessionwise_dict()
        self.partitionwise_dict = self.init_partition_dict()

    def place_clip_output_to_subjectwise_dict(self, clip_output, indices, sessions):
        for index, session in enumerate(sessions):
            subject_id = int(session.split("-")[0].split("P")[1])
            target_range = indices[index]
            self.append_clip_output_to_each_element(str(subject_id), clip_output[index, :, :], target_range)

    def append_clip_output_to_each_element(self, subject_id, clip_output, target_range):

        for emotion, column in zip(self.emotion_dimension, range(clip_output.shape[1])):
            for relative_idx, absolute_idx in enumerate(iter(target_range)):
                self.subjectwise_dict[subject_id][emotion][absolute_idx].append(
                    clip_output[relative_idx, column])

    def average_subjectwise_output(self):
        for subject_id in self.subjectwise_dict:
            for emotion in self.subjectwise_dict[subject_id]:
                length = len(self.subjectwise_dict[subject_id][emotion])

                for index in range(length):
                    self.subjectwise_dict[subject_id][emotion][index] = statistics.mean(
                        self.subjectwise_dict[subject_id][emotion][index])

    def get_partitionwise_dict(self):
        for emotion in self.emotion_dimension:
            for subject_id in self.subjectwise_dict:
                self.partitionwise_dict[emotion].extend(self.subjectwise_dict[subject_id][emotion])

    def get_sessionwise_dict(self):
        self.average_subjectwise_output()

        for subject_id, related_session_length_list in self.length_to_track.items():

            # Use cumsum function to sneakily compute the start indices.
            start_list = np.insert(np.cumsum(self.length_to_track[subject_id]), 0, 0)[:-1]

            for emotion in self.emotion_dimension:
                for index, session_length in enumerate(related_session_length_list):
                    start = start_list[index]
                    end = start + session_length
                    self.sessionwise_dict[subject_id][emotion][index] = self.subjectwise_dict[subject_id][emotion][
                                                                        start:end]

    def init_sessionwise_dict(self):
        sessionwise_dict = {}
        for subject in self.length_to_track:
            intermediate_subjectwise_dict = {key: [] for key in self.emotion_dimension}
            for emotion in self.emotion_dimension:

                for _ in self.length_to_track[subject]:
                    intermediate_subjectwise_dict[emotion].append([])
            sessionwise_dict[subject] = intermediate_subjectwise_dict
        return sessionwise_dict

    def init_subjectwise_dict(self):
        subjectwise_dict = {}
        for subject in self.length_to_track:
            length_sum = np.sum(self.length_to_track[subject])
            subjectwise_list = {emotion: self.init_long_list(length_sum)
                                for emotion in self.emotion_dimension}
            subjectwise_dict[subject] = subjectwise_list
        return subjectwise_dict

    def init_partition_dict(self):
        partitionwise_dict = {key: [] for key in self.emotion_dimension}
        return partitionwise_dict

    @staticmethod
    def init_long_list(length):
        return [[] for _ in range(length)]


class PlotHandler(object):
    r"""
    A class to plot the output-label figures.
    """

    def __init__(self, metrics, emotional_dimension, epoch_result_dict,
                 sessionwise_output_dict, sessionwise_continuous_label_dict,
                 epoch=None, train_mode=None, directory_to_save_plot=None):
        self.metrics = metrics
        self.emotional_dimension = emotional_dimension
        self.epoch_result_dict = epoch_result_dict

        self.epoch = epoch
        self.train_mode = train_mode
        self.directory_to_save_plot = directory_to_save_plot

        self.sessionwise_output_dict = sessionwise_output_dict
        self.sessionwise_continuous_label_dict = sessionwise_continuous_label_dict

    def complete_directory_to_save_plot(self):
        r"""
        Determine the full path to save the plot.
        """
        if self.train_mode:
            exp_folder = "train"
        else:
            exp_folder = "validate"

        if self.epoch is None:
            exp_folder = "test"

        directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder, "epoch_" + str(self.epoch))
        if self.epoch == "test":
            directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder)

        os.makedirs(directory, exist_ok=True)
        return directory

    def reshape_dictionary_to_plot(self, sessionwise_date_dict):
        reshaped_dict = {key: [] for key in sessionwise_date_dict}

        for subject_id, subjectwise_data in sessionwise_date_dict.items():
            # reshaped_dict[subject_id] = []
            number_of_session = len(subjectwise_data[self.emotional_dimension[0]])
            intermediate_list = [{key: [] for key in self.emotional_dimension} for _ in range(number_of_session)]

            for emotion in self.emotional_dimension:
                for session_id, session_data in enumerate(subjectwise_data[emotion]):
                    intermediate_list[session_id][emotion] = session_data

            reshaped_dict[subject_id] = intermediate_list

        return reshaped_dict

    def save_output_vs_continuous_label_plot(self):
        r"""
        Plot the output versus continuous label figures for each session.
        """

        reshaped_sessionwise_output_dict = self.reshape_dictionary_to_plot(self.sessionwise_output_dict)
        reshaped_sessionwise_continuous_label = self.reshape_dictionary_to_plot(self.sessionwise_continuous_label_dict)

        # Determine the full path to save the figures.
        complete_directory = self.complete_directory_to_save_plot()

        # Read the sessionwise data.

        for (subject_id, sessionwise_output), (_, sessionwise_continuous_label) \
                in zip(reshaped_sessionwise_output_dict.items(), reshaped_sessionwise_continuous_label.items()):

            for session_id, (session_output, session_continuous_label) \
                    in enumerate(zip(sessionwise_output, sessionwise_continuous_label)):
                plot_filename = "subject_{}_trial_{}_epoch_{}".format(subject_id, session_id, self.epoch)
                full_plot_filename = os.path.join(complete_directory, plot_filename + ".jpg")

                # Find the y ranges for subplot with better clarity.
                if len(self.emotional_dimension) > 1:
                    ylim_low, ylim_high = [], []
                    for emotion in self.emotional_dimension:
                        ylim_low.append(min(min(session_output[emotion]), min(session_continuous_label[emotion])))
                        ylim_high.append(max(max(session_output[emotion]), max(session_continuous_label[emotion])))
                    ylim_low, ylim_high = min(ylim_low) * 1.15, max(ylim_high) * 1.15
                else:
                    ylim_low, ylim_high = None, None

                self.plot_and_save(full_plot_filename, subject_id, session_id, session_output, session_continuous_label, ylim_low, ylim_high)

    def plot_and_save(self, full_plot_filename, subject_id, session_id, output, continuous_label, ylim_low=None, ylim_high=None):
        fig, ax = plt.subplots(len(self.emotional_dimension), 1)

        for index, emotion in enumerate(self.emotional_dimension):
            result_list = []

            for metric in self.metrics:
                result = self.epoch_result_dict[subject_id][emotion][metric][session_id][0]
                # The pcc usually have two output, one for value and one for confidence. So
                # here we only read and the value and discard the confidence.
                if metric == "pcc":
                    result = self.epoch_result_dict[subject_id][emotion][metric][session_id][0][0]
                result_list.append(result)

            if len(self.emotional_dimension) > 1:
                # Plot the sub-figures, each for one emotional dimension.
                ax[index].plot(output[emotion], "r-", label="Output")
                ax[index].plot(continuous_label[emotion], "g-", label="Label")
                ax[index].set_ylim([ylim_low, ylim_high])
                ax[index].set_xlabel("Sample")
                ax[index].set_ylabel("Value")
                ax[index].legend(loc="upper right", framealpha=0.2)
                ax[index].title.set_text(
                    "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(emotion, *result_list))
            else:
                ax.plot(output[emotion], "r-", label="Output")
                ax.plot(continuous_label[emotion], "g-", label="Label")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Value")
                ax.legend(loc="upper right", framealpha=0.2)
                ax.title.set_text(
                    "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(emotion, *result_list))
        fig.tight_layout()
        plt.savefig(full_plot_filename)
        plt.close()

class PlotHandlerTrial(object):
    r"""
    A class to plot the output-label figures.
    """

    def __init__(self, metrics, emotional_dimension, epoch_result_dict,
                 trialwise_output_dict, trialwise_continuous_label_dict,
                 epoch=None, train_mode=None, directory_to_save_plot=None):
        self.metrics = metrics
        self.emotional_dimension = emotional_dimension
        self.epoch_result_dict = epoch_result_dict

        self.epoch = epoch
        self.train_mode = train_mode
        self.directory_to_save_plot = directory_to_save_plot

        self.trialwise_output_dict = trialwise_output_dict
        self.trialwise_continuous_label_dict = trialwise_continuous_label_dict

    def complete_directory_to_save_plot(self):
        r"""
        Determine the full path to save the plot.
        """
        if self.train_mode:
            exp_folder = "train"
        else:
            exp_folder = "validate"

        if self.epoch is None:
            exp_folder = "test"

        directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder, "epoch_" + str(self.epoch))
        if self.epoch == "test":
            directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder)

        os.makedirs(directory, exist_ok=True)
        return directory

    def save_output_vs_continuous_label_plot(self):
        r"""
        Plot the output versus continuous label figures for each session.
        """

        for (trial, output_record), (_, label_record) in zip(self.trialwise_output_dict.items(), self.trialwise_continuous_label_dict.items()):

            complete_directory = self.complete_directory_to_save_plot()

            subject_id = trial.split("-")[0][1:]
            trial_id = trial.split("-")[1][1:]

            plot_filename = "subject_{}_trial_{}_epoch_{}".format(subject_id, trial_id, self.epoch)
            full_plot_filename = os.path.join(complete_directory, plot_filename + ".jpg")

            # Find the y ranges for subplot with better clarity.
            if len(self.emotional_dimension) > 1:
                ylim_low, ylim_high = [], []
                for emotion in self.emotional_dimension:
                    ylim_low.append(min(min(output_record[emotion]), min(label_record[emotion])))
                    ylim_high.append(max(max(output_record[emotion]), max(label_record[emotion])))
                ylim_low, ylim_high = min(ylim_low) * 1.15, max(ylim_high) * 1.15
            else:
                ylim_low, ylim_high = None, None

            self.plot_and_save(full_plot_filename, trial, output_record, label_record, ylim_low, ylim_high)

    def plot_and_save(self, full_plot_filename, trial, output, continuous_label, ylim_low=None, ylim_high=None):
        fig, ax = plt.subplots(len(self.emotional_dimension), 1)

        for index, emotion in enumerate(self.emotional_dimension):
            result_list = []

            for metric in self.metrics:
                result = self.epoch_result_dict[trial][emotion][metric][0]
                # The pcc usually have two output, one for value and one for confidence. So
                # here we only read and the value and discard the confidence.
                if metric == "pcc":
                    result = self.epoch_result_dict[trial][emotion][metric][0][0]
                result_list.append(result)

            if len(self.emotional_dimension) > 1:
                # Plot the sub-figures, each for one emotional dimension.
                ax[index].plot(output[emotion], "r-", label="Output")
                ax[index].plot(continuous_label[emotion], "g-", label="Label")
                ax[index].set_ylim([ylim_low, ylim_high])
                ax[index].set_xlabel("Sample")
                ax[index].set_ylabel("Value")
                ax[index].legend(loc="upper right", framealpha=0.2)
                ax[index].title.set_text(
                    "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(emotion, *result_list))
            else:
                ax.plot(output[emotion], "r-", label="Output")
                ax.plot(continuous_label[emotion], "g-", label="Label")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Value")
                ax.legend(loc="upper right", framealpha=0.2)
                ax.title.set_text(
                    "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(emotion, *result_list))
        fig.tight_layout()
        plt.savefig(full_plot_filename)
        plt.close()

