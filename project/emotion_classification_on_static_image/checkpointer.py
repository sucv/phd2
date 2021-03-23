from base.checkpointer import GenericCheckpointer
from base.utils import load_single_pkl, save_pkl_file

import os
import time
import copy

import pandas as pd


class Checkpointer(GenericCheckpointer):
    def __init__(self, keys, path, trainer, parameter_controller, resume):
        super().__init__(keys, path, trainer, parameter_controller, resume)
        self.columns = []

    def save_log_to_csv(self, epoch):
        num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])

        csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                       self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1],
                       self.trainer.validate_losses[-1], self.trainer.train_accuracies[-1], self.trainer.validate_accuracies[-1]]

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)

    def init_csv_logger(self):
        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr',
                        'tr_loss', 'val_loss', 'tr_acc', 'val_acc']

        df = pd.DataFrame(columns=self.columns)
        self.trainer.csv_filename = self.trainer.model_path[:-4] + ".csv"
        df.to_csv(self.trainer.csv_filename, index=False)

    def read_checkpoint(self):
            self.init_csv_logger()
