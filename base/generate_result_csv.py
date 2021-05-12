import pandas as pd
from pathlib import Path
import numpy as np
import ast
import os
import sys
import argparse


def csv_generator(path_to_read_result, num_folds=10):

    fold_wise_csv_filename = "training_logs.csv"
    num_folds = num_folds
    metrics = ["rmse", "pcc", "ccc"]
    result = np.zeros((len(metrics), num_folds + 2))


    for path in Path(path_to_read_result).rglob(fold_wise_csv_filename):
        fold = int(path.parts[6])

        df = pd.read_csv(path)
        last_row = df.iloc[-1, :]

        rmse = float(ast.literal_eval(last_row.iloc[2])[0])
        pcc = float(last_row.iloc[4])
        ccc = float(ast.literal_eval(last_row.iloc[7])[0])

        result[0, fold] = rmse
        result[1, fold] = pcc
        result[2, fold] = ccc

    result[:, -2] = np.mean(result[:, :-2], axis=1)
    result[:, -1] = np.std(result[:, :-2], axis=1)

    result_csv = pd.DataFrame(result, columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'mean', 'std'], index=metrics)

    csv_filename = os.path.join(path_to_read_result, "result.csv")
    result_csv.to_csv(csv_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-path_to_read_result', default='/home/zhangsu/phd2/save/emotion_video_1d_only_reg_v_eeg_psd_bs_2', type=str,
                            help='The root directory of the dataset.')
    args = parser.parse_args()
    csv_generator(args.path_to_read_result)
