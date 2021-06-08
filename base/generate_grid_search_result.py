from base.generate_result_csv import csv_generator
from xlsxwriter.workbook import Workbook
import os
import pandas as pd

# path_template = "/home/zhangsu/phd2/save/nscc_check/nscc_emo_kd_1d_only_reg_v_eeg_psd_sub_ind_1e5_ccc_weight_{arg1}_kd_weight_{arg2}_{arg3:s}_1"
# path_template = "/home/zhangsu/phd2/save/colab_check/colab_emo_kd_1d_only_reg_v_eeg_psd_trial_128x2_lr_1e5_ccc_weight_{arg1}_kd_weight_{arg2}_{arg3:s}_1"
path_template = "/home/zhangsu/phd2/save/group_check/group_emo_kd_1d_only_reg_v_eeg_psd_trial_1e5_ccc_weight_{arg1}_kd_weight_{arg2}_{arg3:s}_1"
# arg1s = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
# arg1s = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
arg1s = [1]
arg2s = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# arg2s = [0.0, 0.1, 0.2, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
# arg2s = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.35, 2.0, 3.0]
arg3s = ["l1"]

multi_index_list = []

overall_xls_path = "/home/zhangsu/phd2/save/nscc_check/overall_result_group_trial.xlsx"
writer = pd.ExcelWriter(overall_xls_path, engine='xlsxwriter')

num_rows = 11
num_columns = 18

for sheet_index, loss_func in enumerate(arg3s):

    for i, ccc_weight in enumerate(arg1s):
        for j, kd_weight in enumerate(arg2s):
            path = path_template.format(arg1=ccc_weight, arg2=kd_weight, arg3=loss_func)

            setting = path.split(os.sep)[-1]
            csv_generator(path_to_read_result=path, num_folds=10)

            output_file = os.path.join(path, "result.csv")
            csv_df = pd.read_csv(output_file)

            startcol = j * num_columns
            startrow = i * num_rows

            csv_df.to_excel(writer, sheet_name=str(loss_func), startcol=startcol, startrow=startrow+1)

            worksheet = writer.sheets[str(loss_func)]
            worksheet.write(startrow, startcol, setting)

writer.save()




