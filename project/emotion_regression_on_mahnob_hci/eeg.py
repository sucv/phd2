from base.eeg import GenericEegController

import mne

class EegMahnob(GenericEegController):
    def __init__(self, filename, buffer=2):
        self.filename = filename
        self.buffer = buffer
        self.frequency = 256
        self.raw_data = self.read_data()
        self.channel_slice = self.get_channel_slice()
        self.channel_type_dictionary = self.get_channel_type_dictionary()
        self.raw_data = self.set_channel_types_from_dictionary()
        self.crop_range = self.get_crop_range_in_second()
        self.cropped_raw_data = self.crop_data()

        self.cropped_eeg_data = self.get_eeg_data()

        self.average_referenced_data = self.average_reference()
        self.filtered_data = self.filter_eeg_data()

    def filter_eeg_data(self):
        r"""
        Filter the eeg signal using lowpass and highpass filter.
        :return: (mne object), the filtered eeg signal.
        """
        filtered_eeg_data = self.average_referenced_data.filter(l_freq=0.3, h_freq=45)
        return filtered_eeg_data[:][0].T

    def average_reference(self):
        average_referenced_data = self.cropped_eeg_data.copy().load_data().set_eeg_reference()
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

    def get_channel_type_dictionary(self):
        r"""
        Generate a dictionary where the key is the channel names, and the value
            is the modality name (such as eeg, ecg, eog, etc...)
        :return: (dict), the dictionary of channel names to modality name.
        """
        channel_type_dictionary = {}
        for modal, slicing in self.channel_slice.items():
            channel_type_dictionary.update({channel: modal
                                            for channel in self.raw_data.ch_names[
                                                self.channel_slice[modal]]})

        return channel_type_dictionary

    def set_channel_types_from_dictionary(self):
        r"""
        Set the channel types of the raw data according to a dictionary. I did this
            in order to call the automatic EOG, ECG remover. But it currently failed. Need to check.
        :return:
        """
        data = self.raw_data.set_channel_types(self.channel_type_dictionary)
        return data

    def get_crop_range_in_second(self):
        r"""
        Assign the stimulated time interval for cropping.
        :return: (list), the list containing the time interval.
        """
        crop_range = [[30., self.raw_data.times.max() - 30 + self.buffer]]
        return crop_range

    def crop_data(self):
        r"""
        Crop the signal so that only the stimulated parts are preserved.
        :return: (mne object), the cropped data.
        """
        cropped_data = []
        for index, (start, end) in enumerate(self.crop_range):

            if index == 0:
                cropped_data = self.raw_data.copy().crop(tmin=start, tmax=end)
            else:
                cropped_data.append(self.raw_data.copy().crop(tmin=start, tmax=end))

        return cropped_data

    def get_eeg_data(self):
        r"""
        Get only the eeg data from the raw data.
        :return: (mne object), the eeg signal.
        """
        eeg_data = self.cropped_raw_data.copy().pick_types(eeg=True)
        return eeg_data