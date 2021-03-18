from base.checkpointer import GenericCheckpointer
from base.utils import load_single_pkl, save_pkl_file

import os
import time
import copy

import pandas as pd


class Checkpointer(GenericCheckpointer):
    def __init__(self, keys, path, trainer):
        super().__init__(keys, path, trainer)
        self.columns = []

    def load_checkpoint(self):
        # If checkpoint file exists, then read it.
        if os.path.isfile(self.path):
            print("Loading checkpoint. Are you sure it is intended?")
            self.checkpoint = {**self.checkpoint, **load_single_pkl(self.path)}
            print("Checkpoint loaded!")
            print("Fitting completed?", str(self.checkpoint['fit_finished']))
            print("Start epoch:", str(self.checkpoint['start_epoch']))
        else:
            print("Checkpoint not exists, initialized one instead.")
            self.checkpoint = self.checkpoint

    def save_checkpoint(self, epoch, path):
        self.checkpoint['time_fit_start'] = self.trainer.time_fit_start
        self.checkpoint['start_epoch'] = epoch + 1
        self.checkpoint['early_stopping_counter'] = self.trainer.early_stopping_counter
        self.checkpoint['best_epoch_info'] = self.trainer.best_epoch_info
        self.checkpoint['combined_record_dict'] = self.trainer.combined_record_dict
        self.checkpoint['train_losses'] = self.trainer.train_losses
        self.checkpoint['validate_losses'] = self.trainer.validate_losses
        self.checkpoint['csv_filename'] = self.trainer.csv_filename
        self.checkpoint['optimizer'] = self.trainer.optimizer
        self.checkpoint['scheduler'] = self.trainer.scheduler
        self.checkpoint['param_control'] = self.trainer.parameter_control
        self.checkpoint['current_model_weights'] = copy.deepcopy(self.trainer.model.state_dict())
        self.checkpoint['fit_finished'] = False
        self.checkpoint['fold_finished'] = False

        if path:
            print("Saving checkpoint.")
            save_pkl_file(path, "checkpoint.pkl", self.checkpoint)
            print("Checkpoint saved.")

    def save_log_to_csv(self, epoch, mean_train_record, mean_validate_record):
        num_layers_to_update = len(self.trainer.optimizer.param_groups[0]['params'])

        csv_records = [time.time(), epoch, int(self.trainer.best_epoch_info['epoch']), num_layers_to_update,
                       self.trainer.optimizer.param_groups[0]['lr'], self.trainer.train_losses[-1],
                       self.trainer.validate_loss[-1]]

        if self.trainer.head == "single-headed":
            if self.trainer.train_emotion == "arousal":
                csv_records.extend([
                    mean_train_record['Arousal']['rmse'][0], mean_train_record['Arousal']['pcc'][0][0],
                    mean_train_record['Arousal']['pcc'][0][1], mean_train_record['Arousal']['ccc'][0],
                    mean_validate_record['Arousal']['rmse'][0], mean_validate_record['Arousal']['pcc'][0][0],
                    mean_validate_record['Arousal']['pcc'][0][1], mean_validate_record['Arousal']['ccc'][0]])
            else:
                csv_records.extend([
                    mean_train_record['Valence']['rmse'][0], mean_train_record['Valence']['pcc'][0][0],
                    mean_train_record['Valence']['pcc'][0][1], mean_train_record['Valence']['ccc'][0],
                    mean_validate_record['Valence']['rmse'][0], mean_validate_record['Valence']['pcc'][0][0],
                    mean_validate_record['Valence']['pcc'][0][1], mean_validate_record['Valence']['ccc'][0]])
        else:
            csv_records.extend([
                                   mean_train_record['Arousal']['rmse'][0], mean_train_record['Arousal']['pcc'][0][0],
                                   mean_train_record['Arousal']['pcc'][0][1], mean_train_record['Arousal']['ccc'][0],
                                   mean_validate_record['Arousal']['rmse'][0], mean_validate_record['Arousal']['pcc'][0][0],
                                   mean_validate_record['Arousal']['pcc'][0][1], mean_validate_record['Arousal']['ccc'][0]] +
                               [
                                   mean_train_record['Valence']['rmse'][0], mean_train_record['Valence']['pcc'][0][0],
                                   mean_train_record['Valence']['pcc'][0][1], mean_train_record['Valence']['ccc'][0],
                                   mean_validate_record['Valence']['rmse'][0], mean_validate_record['Valence']['pcc'][0][0],
                                   mean_validate_record['Valence']['pcc'][0][1], mean_validate_record['Valence']['ccc'][0]])

        row_df = pd.DataFrame(data=csv_records)
        row_df.T.to_csv(self.trainer.csv_filename, mode='a', index=False, header=False)

    def init_csv_logger(self):
        self.columns = ['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr', 'plateau_count',
                        'tr_loss', 'val_loss']

        if self.trainer.head == "single-headed":
            if self.trainer.train_emotion == "arousal":
                self.columns.extend(['tr_rmse_a', 'tr_pcc_a_v', 'tr_pcc_a_conf', 'tr_ccc_a',
                                     'val_rmse_a', 'val_pcc_a_v', 'val_pcc_a_conf', 'val_ccc_a'])
            else:
                self.columns.extend(['tr_rmse_v', 'tr_pcc_v_v', 'tr_pcc_v_conf', 'tr_ccc_v',
                                     'val_rmse_v', 'val_pcc_v_v', 'val_pcc_v_conf', 'val_ccc_v'])
        else:
            self.columns.extend(['tr_rmse_a', 'tr_pcc_a_v', 'tr_pcc_a_conf', 'tr_ccc_a',
                                 'val_rmse_a', 'val_pcc_a_v', 'val_pcc_a_conf', 'val_ccc_a'] \
                                + ['tr_rmse_v', 'tr_pcc_v_v', 'tr_pcc_v_conf', 'tr_ccc_v',
                                   'val_rmse_v', 'val_pcc_v_v', 'val_pcc_v_conf', 'val_ccc_v'])

        df = pd.DataFrame(columns=self.columns)
        csv_filename = self.trainer.model_path[:-4] + ".csv"
        df.to_csv(csv_filename, index=False)

    def read_checkpoint(self):
        if self.checkpoint['current_model_weights']:
            self.trainer.time_fit_start = self.checkpoint['time_fit_start']
            self.trainer.csv_filename = self.checkpoint['csv_filename']
            self.trainer.start_epoch = self.checkpoint['start_epoch']
            self.trainer.early_stopping_counter = self.checkpoint['early_stopping_counter']
            self.trainer.best_epoch_info = self.checkpoint['best_epoch_info']
            self.trainer.combined_train_record_dict = self.checkpoint['combined_train_record_dict']
            self.trainer.combined_validate_record_dict = self.checkpoint['combined_validate_record_dict']
            self.trainer.train_losses = self.checkpoint['train_losses']
            self.trainer.validate_losses = self.checkpoint['validate_losses']
            self.trainer.current_model_weights = self.checkpoint['current_model_weights']
            self.trainer.optimizer = self.checkpoint['optimizer']
            self.trainer.scheduler = self.checkpoint['scheduler']
            self.trainer.parameter_control = self.checkpoint['param_control']
        else:
            self.init_csv_logger()
