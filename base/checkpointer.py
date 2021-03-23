class GenericCheckpointer(object):
    def __init__(self, keys, path, trainer, parameter_controller, resume):
        self.checkpoint = {key: False for key in keys}
        self.path = path
        self.trainer = trainer
        self.parameter_controller = parameter_controller
        self.resume = resume



