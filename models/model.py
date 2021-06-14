from models.arcface_model import Backbone, Backbone_Eeg
from models.temporal_convolutional_model import TemporalConvNet
from models.eeg_net import EEGNet
import os
import torch
from torch import nn

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_res50(nn.Module):
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512, fix_backbone=True):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                        Dropout(0.4),
                                                        Flatten(),
                                                        Linear(embedding_dim * 5 * 5, embedding_dim),
                                                        BatchNorm1d(embedding_dim))

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

            if fix_backbone:
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


class my_res50_tempool(nn.Module):
    def __init__(self, backbone_mode="ir", state_dict_name='', embedding_dim=512, input_channels=5, output_dim=1,
                 root_dir='', use_pretrained=False):
        super().__init__()

        self.backbone = my_res50(
            input_channels=input_channels, mode=backbone_mode, root_dir=root_dir,
            use_pretrained=use_pretrained, state_dict_name=state_dict_name, ).backbone

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

class my_eegnet_temporal(nn.Module):
    def __init__(self, num_channels=60, num_samples=151, dropout_rate=0.5, kernel_length=64, kernel_length2=16, F1=8,
                 D=2, F2=16, num_classes=3):

        super().__init__()
        num_inputs = num_samples // 32 * F2
        self.spatial = EEGNet(num_channels=num_channels, num_samples=num_samples, dropout_rate=dropout_rate,
                              kernel_length=kernel_length, kernel_length2=kernel_length2, F1=F1, F2=F2, D=D).blocks

        self.logits = nn.Linear(in_features=256, out_features=num_classes)
        self.temporal_pooling = nn.AvgPool1d(kernel_size=96)

    def forward(self, x):
        num_batches, _, channel, sample = x.shape
        x = self.spatial(x)
        x = x.view(num_batches, -1)
        x = self.logits(x)
        return x

class my_temporal(nn.Module):
    def __init__(self, model_name, num_inputs=192, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5, cnn1d_dropout_rate=0.1,
                 embedding_dim=256, hidden_dim=128, lstm_dropout_rate=0.5, bidirectional=True, output_dim=1):
        super().__init__()
        self.model_name = model_name
        if "1d" in model_name:
            self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                       kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)
            self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

        elif "lstm" in model_name:
            self.temporal = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout_rate)
            input_dim = hidden_dim
            if bidirectional:
                input_dim = hidden_dim * 2

            self.regressor = nn.Linear(input_dim, output_dim)


    def forward(self, x, test=None):
        features = {}
        if "lstm_only" in self.model_name:
            x = x.transpose(1, 2).contiguous()
            x, _ = self.temporal(x)
            x = x.contiguous()
        else:
            x = self.temporal(x).transpose(1, 2).contiguous()
        batch, time_step, temporal_feature_dim = x.shape
        features['temporal'] = x
        if test is not None:
            x = test
        x = x.view(-1, temporal_feature_dim)
        x = self.regressor(x)
        x = x.view(batch, time_step, 1)
        return x, features


class my_test(nn.Module):
    def __init__(self):
        super().__init__()

        path = "/home/zhangsu/phd2/load/trained_2d1d_frame/2d1d_v_1.pth"
        state_dict = torch.load(path, map_location='cpu')
        new_dict = {}

        for key, value in state_dict.items():
            if "regressor" in key:
                key = key[10:]
                new_dict[key] = value

        self.regressor = nn.Linear(128, 1)
        # self.regressor.load_state_dict(new_dict)
        # for param in self.regressor.parameters():
        #     param.requires_grad = False


    def forward(self, x, test=None):
        batch, step, feat = x.shape
        x = x.view(-1, feat)
        x = self.regressor(x)
        x = x.view(batch, step, 1)
        return x


class my_eeglstm(nn.Module):
    def __init__(self, num_channels=60, num_samples=151, dropout_rate=0.5, kernel_length=64, kernel_length2=16, F1=8,
                 D=2, F2=16, embedding_dim=256, hidden_dim=128, output_dim=1, lstm_dropout_rate=0.5):
        super().__init__()
        num_inputs = num_samples // 32 * F2
        self.spatial = EEGNet(num_channels=num_channels, num_samples=num_samples, dropout_rate=dropout_rate,
                              kernel_length=kernel_length, kernel_length2=kernel_length2, F1=F1, F2=F2, D=D).blocks

        self.temporal = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=lstm_dropout_rate)

        self.regressor = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        batch, time_step, _, electrode, sample = x.shape
        x = x.view(-1, 1, electrode, sample)
        x = self.spatial(x)
        x = x.view(batch, time_step, -1)

        x, _ = self.temporal(x)
        x = x.contiguous()
        _, _, temporal_feature_dim = x.shape
        x = x.view(-1, temporal_feature_dim)
        x = self.regressor(x)
        x = x.view(batch, time_step, 1)
        return x

class my_eeg1d(nn.Module):
    def __init__(self, num_channels=60, num_samples=151,
                 dropout_rate=0.5, kernel_length=64, kernel_length2=16, F1=8, D=2, F2=16,
                 cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5, cnn1d_dropout_rate=0.1, output_dim=1):
        super().__init__()

        num_inputs = num_samples // 32 * F2
        self.spatial = EEGNet(num_channels=num_channels, num_samples=num_samples, dropout_rate=dropout_rate,
                         kernel_length=kernel_length, kernel_length2=kernel_length2, F1=F1, F2=F2, D=D).blocks

        self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                   kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)

        self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

    def forward(self, x):
        batch, time_step, _, electrode, sample = x.shape
        x = x.view(-1, 1, electrode, sample)
        x = self.spatial(x)
        x = x.view(batch, time_step, -1).transpose(2, 1).contiguous()

        x = self.temporal(x).transpose(1, 2).contiguous()
        _, _, temporal_feature_dim = x.shape
        x = x.view(-1, temporal_feature_dim)
        x = self.regressor(x)
        x = x.view(batch, time_step, 1)
        return x


class my_2d1d(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None,
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
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

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
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size,
            dropout=self.dropout)
        self.regressor = nn.Linear(self.embedding_dim // 4, self.output_dim)
        # self.regressor = Sequential(
        #     BatchNorm1d(self.embedding_dim // 4),
        #     Dropout(0.4),
        #     Linear(self.embedding_dim // 4, self.output_dim))

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
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

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
