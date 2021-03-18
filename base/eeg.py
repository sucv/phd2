class GenericEegController(object):

    def filter_eeg_data(self):
        raise NotImplementedError

    def average_reference(self):
        raise NotImplementedError

    def read_data(self):
        raise NotImplementedError

    def get_crop_range_in_second(self):
        raise NotImplementedError

    def crop_data(self):
        raise NotImplementedError

    def get_eeg_data(self):
        raise NotImplementedError