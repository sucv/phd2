from base.checkpointer import GenericCheckpointer

import os
import time

import pandas as pd
import numpy as np


class Checkpointer(GenericCheckpointer):
    def __init__(self, path, trainer, parameter_controller, resume):
        super().__init__(path, trainer, parameter_controller, resume)
        self.columns = []

    def save_log_to_csv(self, epoch=None):
        np.set_printoptions(suppress=True)
        num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])

        if epoch is None:
            csv_records = ["Test results: ", "accuracy: ", self.trainer.test_accuracy, self.trainer.test_confusion_matrix]
        else:
            csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                           self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1],
                           self.trainer.validate_losses[-1], self.trainer.train_accuracies[-1], self.trainer.validate_accuracies[-1],
                           self.trainer.train_confusion_matrices[-1], self.trainer.validate_confusion_matrices[-1]]

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)
        np.set_printoptions()

    def init_csv_logger(self, args, config):

        self.trainer.csv_filename = os.path.join(self.trainer.save_path, "training_logs.csv")

        # Record the arguments.
        arguments_dict = vars(args)
        arguments_dict = pd.json_normalize(arguments_dict, sep='_')

        df_args = pd.DataFrame(data=arguments_dict)
        df_args.to_csv(self.trainer.csv_filename, index=False)

        config = pd.json_normalize(config, sep='_')
        df_config = pd.DataFrame(data=config)
        df_config.to_csv(self.trainer.csv_filename, mode='a', index=False)

        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr',
                        'tr_loss', 'val_loss', 'tr_acc', 'val_acc', 'tr_conf_mat', 'val_conf_mat']

        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.trainer.csv_filename, mode='a', index=False)
