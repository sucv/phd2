from base.generate_result_csv import csv_generator
from xlsxwriter.workbook import Workbook
import os
import pandas as pd

path_template = "/home/zhangsu/phd2/save/group_emo_kd_subind_1d_only_reg_v_eeg_psd_grid_ccc_weight_{arg1}_kd_weight_{arg2:.1f}_{arg3:s}_1"

# arg1s = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
# arg1s = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
arg1s = [0.005, 0.01, 0.015, 0.02, 0.03, 0.035, 0.04, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085]
arg2s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
arg3s = ["l1"]

multi_index_list = []

overall_xls_path = "/home/zhangsu/phd2/save/overall_result.xlsx"
writer = pd.ExcelWriter(overall_xls_path, engine='xlsxwriter')

num_rows = 11
num_columns = 16

for sheet_index, loss_func in enumerate(arg3s):

    for i, ccc_weight in enumerate(arg1s):
        for j, kd_weight in enumerate(arg2s):
            path = path_template.format(arg1=ccc_weight, arg2=kd_weight, arg3=loss_func)

            setting = path.split(os.sep)[-1]
            csv_generator(path_to_read_result=path)

            output_file = os.path.join(path, "result.csv")
            csv_df = pd.read_csv(output_file)

            startcol = j * num_columns
            startrow = i * num_rows

            csv_df.to_excel(writer, sheet_name=str(loss_func), startcol=startcol, startrow=startrow+1)

            worksheet = writer.sheets[str(loss_func)]
            worksheet.write(startrow, startcol, setting)

writer.save()




