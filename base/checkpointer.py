from base.utils import load_single_pkl, save_pkl_file

import os


class GenericCheckpointer(object):
    def __init__(self, path, trainer, parameter_controller, resume):
        self.checkpoint = {}
        self.path = path
        self.trainer = trainer
        self.parameter_controller = parameter_controller
        self.resume = resume

    def load_checkpoint(self):
        # If checkpoint file exists, then read it.
        if os.path.isfile(self.path):
            print("Loading checkpoint. Are you sure it is intended?")
            self.checkpoint = {**self.checkpoint, **load_single_pkl(self.path)}
            print("Checkpoint loaded!")

            self.trainer = self.checkpoint['trainer']
            self.trainer.resume = True
            self.parameter_controller = self.checkpoint['param_control']
            self.parameter_controller.trainer = self.trainer
        else:
            raise ValueError("Checkpoint not exists!!")
        return self.trainer, self.parameter_controller

    def save_checkpoint(self, trainer, parameter_controller, path):
        self.checkpoint['trainer'] = trainer
        self.checkpoint['param_control'] = parameter_controller

        if path:
            print("Saving checkpoint.")
            save_pkl_file(path, "checkpoint.pkl", self.checkpoint)
            print("Checkpoint saved.")



