from models.arcface_model import Backbone, Backbone_Eeg
from models.temporal_convolutional_model import TemporalConvNet

import os
import torch
from torch import nn

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_res50(nn.Module):
    def __init__(self, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512):
        super().__init__()
        self.backbone = Backbone(num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.logits(x)
        return x


class my_res50_eeg(nn.Module):
    def __init__(self, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512):
        super().__init__()
        self.backbone = Backbone_Eeg(num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = True

        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.logits(x)
        return x


class my_res50_tempool(nn.Module):
    def __init__(self, backbone_mode="ir", embedding_dim=512, output_dim=1, root_dir=''):
        super().__init__()

        self.backbone = my_res50_eeg(mode=backbone_mode, root_dir=root_dir, use_pretrained=False).backbone
        self.logits = nn.Linear(in_features=embedding_dim, out_features=output_dim)
        self.temporal_pooling = nn.AvgPool1d(kernel_size=96)


    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.backbone(x)
        x = self.logits(x)
        _, output_dim = x.shape
        x = x.view(num_batches, length, output_dim).transpose(1, 2).contiguous()
        x = self.temporal_pooling(x)
        x = x.transpose(1, 2).contiguous().squeeze()
        return x


class my_2d1d(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality='frame', embedding_dim=512, channels=None,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir=''):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dropout = dropout

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        if 'frame' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        elif 'eeg_image' in self.modality:
            spatial = my_res50_eeg(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3)

            if fold is not None:
                path = os.path.join(self.root_dir, self.backbone_state_dict + "_" + str(fold) + ".pth")

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        else:
            raise ValueError("Unsupported modality!")

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size, dropout=self.dropout)
        self.regressor = nn.Linear(self.embedding_dim // 4, self.output_dim)

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x


class my_2dlstm(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality='frame', embedding_dim=512, hidden_dim=256,
                 output_dim=1, dropout=0.5, root_dir=''):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        if 'frame' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        elif 'eeg_image' in self.modality:
            spatial = my_res50_eeg(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False,
                                   num_classes=3)

            if fold is not None:
                path = os.path.join(self.root_dir, self.backbone_state_dict + "_" + str(fold) + ".pth")

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        else:
            raise ValueError("Unsupported modality!")

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=self.dropout)
        self.regressor = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).contiguous()
        x, _ = self.temporal(x)
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x
