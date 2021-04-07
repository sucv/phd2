import torch

import os
from collections import OrderedDict

model_dict_path = "/home/zhangsu/phd2/load/mahnob_reg_v_9.pth"
model_dict_newname = "/home/zhangsu/phd2/load/mahnob_reg_v_9_new.pth"

state_dict = torch.load(model_dict_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k[:3] == "fc.":
        k_new = "logits." + k[3:]
    else:
        k_new = k
    new_state_dict[k_new] = v

torch.save(new_state_dict, model_dict_path)
print(0)