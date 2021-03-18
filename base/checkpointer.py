class GenericCheckpointer(object):
    def __init__(self, keys, path, trainer):
        self.checkpoint = {key: False for key in keys}
        self.path = path
        self.trainer = trainer

