from models.arcface_model import Backbone

import os
import torch
from torch import nn

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_res50(nn.Module):
    def __init__(self, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir"):
        super().__init__()
        self.backbone = Backbone(num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(0.4),
                                       Flatten(),
                                       Linear(512 * 5 * 5, 512),
                                       BatchNorm1d(512))

        self.logits = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.logits(x)
        return x


