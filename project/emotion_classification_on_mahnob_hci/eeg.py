from base.eeg import GenericEegController, bandpower_multiple, gen_images

import mne
import numpy as np


class EegMahnob(GenericEegController):
    def __init__(self, filename, buffer=2, electrode_2d_pos=None, eeg_image_size=40):
        self.filename = filename
        self.buffer = buffer
        self.frequency = 256
        self.window_sec = 0.75
        self.eeg_image_size = eeg_image_size
        self.step = int(0.25 * self.frequency)
        self.electrode_2d_pos = electrode_2d_pos
        self.interest_bands = self.set_interest_bands()
        self.channel_slice = self.get_channel_slice()

        self.filtered_raw_eeg, self.eeg_images = self.preprocessing()


    def calculate_psd(self, data):
        data_np = data[:][0]
        power_spectram_densities = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            psd = bandpower_multiple(data_np[:, start:end], sampling_frequence=self.frequency, band_sequence=self.interest_bands, window_sec=self.window_sec, relative=True)
            power_spectram_densities.extend(psd)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        power_spectram_densities = np.asarray(power_spectram_densities)
        return power_spectram_densities

    def preprocessing(self):
        raw_data = self.read_data()
        channel_type_dictionary = self.get_channel_type_dictionary(raw_data)
        raw_data = self.set_channel_types_from_dictionary(raw_data, channel_type_dictionary)
        crop_range = self.get_crop_range_in_second(raw_data)
        cropped_raw_data = self.crop_data(raw_data, crop_range)
        cropped_eeg_raw_data = self.get_eeg_data(cropped_raw_data)
        average_referenced_data = self.average_reference(cropped_eeg_raw_data)
        filtered_data_np = self.filter_eeg_data(average_referenced_data)
        power_spectram_densities = self.calculate_psd(average_referenced_data)
        eeg_images = self.create_eeg_image(power_spectram_densities)

        return filtered_data_np, eeg_images

    def create_eeg_image(self, power_spectram_densities):
        power_spectram_densities = power_spectram_densities[np.newaxis, :]
        eeg_images = []
        for i in range(power_spectram_densities.shape[1] // 160):
            eeg_image = gen_images(self.electrode_2d_pos, power_spectram_densities[:, i * 160 : (i+1) * 160], self.eeg_image_size, normalize=True)
            eeg_images.append(eeg_image)
        eeg_images = np.vstack(eeg_images)
        eeg_images = np.transpose(eeg_images, (0, 2, 3, 1))
        return eeg_images

    @staticmethod
    def set_interest_bands():
        interest_bands = [(0.3, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
        return interest_bands

    @staticmethod
    def filter_eeg_data(data):
        r"""
        Filter the eeg signal using lowpass and highpass filter.
        :return: (mne object), the filtered eeg signal.
        """
        filtered_eeg_data = data.filter(l_freq=0.3, h_freq=45)
        return filtered_eeg_data[:][0].T

    @staticmethod
    def average_reference(data):
        average_referenced_data = data.copy().load_data().set_eeg_reference()
        return average_referenced_data

    def read_data(self):
        r"""
        Load the bdf data using mne API.
        :return: (mne object), the raw signal containing different channels.
        """
        filename = self.filename

        if filename.endswith(".bdf"):
            raw_data = mne.io.read_raw_bdf(filename)

        return raw_data

    def get_channel_slice(self):
        r"""
        Assign a tag to each channel according to the dataset paradigm.
        :return:
        """
        channel_slice = {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)}
        return channel_slice

    def get_channel_type_dictionary(self, data):
        r"""
        Generate a dictionary where the key is the channel names, and the value
            is the modality name (such as eeg, ecg, eog, etc...)
        :return: (dict), the dictionary of channel names to modality name.
        """
        channel_type_dictionary = {}
        for modal, slicing in self.channel_slice.items():
            channel_type_dictionary.update({channel: modal
                                            for channel in data.ch_names[
                                                self.channel_slice[modal]]})

        return channel_type_dictionary

    @staticmethod
    def set_channel_types_from_dictionary(data, channel_type_dictionary):
        r"""
        Set the channel types of the raw data according to a dictionary. I did this
            in order to call the automatic EOG, ECG remover. But it currently failed. Need to check.
        :return:
        """
        data = data.set_channel_types(channel_type_dictionary)
        return data

    def get_crop_range_in_second(self, data):
        r"""
        Assign the stimulated time interval for cropping.
        :return: (list), the list containing the time interval.
        """
        crop_range = [[30. - 0.25, data.times.max() - 30 + self.buffer]]
        return crop_range

    @staticmethod
    def crop_data(data, crop_range):
        r"""
        Crop the signal so that only the stimulated parts are preserved.
        :return: (mne object), the cropped data.
        """
        cropped_data = []
        for index, (start, end) in enumerate(crop_range):

            if index == 0:
                cropped_data = data.copy().crop(tmin=start, tmax=end)
            else:
                cropped_data.append(data.copy().crop(tmin=start, tmax=end))

        return cropped_data

    @staticmethod
    def get_eeg_data(data):
        r"""
        Get only the eeg data from the raw data.
        :return: (mne object), the eeg signal.
        """
        eeg_data = data.copy().pick_types(eeg=True)
        return eeg_data