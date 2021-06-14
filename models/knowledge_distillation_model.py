from models.arcface_model import Backbone

from models.model import my_res50, my_2d1d, my_2dlstm
from models.temporal_convolutional_model import TemporalConvNet

import os

import torch
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module, AvgPool1d


class kd_res50(Module):
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512, folder='', role="teacher"):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.state_dict_name = state_dict_name
        self.root_dir = root_dir
        self.folder = folder
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.role = role

        self.logits = Linear(in_features=self.embedding_dim, out_features=self.num_classes)

    def init(self, fold):
        path = os.path.join(self.root_dir, self.state_dict_name + ".pth")

        if self.role == "teacher":
            spatial = my_res50(mode=self.mode, root_dir=self.root_dir, use_pretrained=False)

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)

        else:
            spatial = my_res50(mode=self.mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

            if fold is not None:
                path = os.path.join(self.root_dir, self.folder, self.state_dict_name + "_" + str(fold) + ".pth")

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.backbone = spatial.backbone

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.backbone(x)
        return x


class kd_2d1d(my_2d1d):
    def __init__(self, backbone_state_dict='', backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None,
                 output_dim=1, kernel_size=5, dropout=0.1, num_logits=192, root_dir='', folder='2d1d', knowledge=[],
                 role='student'):
        super().__init__(backbone_state_dict=backbone_state_dict, backbone_mode=backbone_mode, modality=modality,
                         embedding_dim=embedding_dim, channels=channels,
                         output_dim=output_dim, kernel_size=kernel_size, dropout=dropout, root_dir=root_dir)

        self.model = my_2d1d(backbone_state_dict='', backbone_mode=backbone_mode, modality=modality,
                             output_dim=output_dim, kernel_size=kernel_size, dropout=dropout, root_dir=root_dir)

        self.folder = folder
        self.role = role
        self.knowledge = knowledge

        self.fc = Linear(embedding_dim, num_logits)

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.folder, self.backbone_state_dict + ".pth")
        if fold is not None:
            path = os.path.join(self.root_dir, self.folder, self.backbone_state_dict + "_" + str(fold) + ".pth")

        if 'frame' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)

        elif 'eeg_image' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

        else:
            raise ValueError("Unsupported modality!")

        if self.role == "student":
            state_dict = torch.load(path, map_location='cpu')
            spatial.load_state_dict(state_dict)
            for param in spatial.parameters():
                param.requires_grad = False

        self.model.spatial = spatial.backbone
        self.model.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size,
            dropout=self.dropout)
        self.model.regressor = Linear(self.embedding_dim // 4, self.output_dim)

        if self.role == "teacher":
            state_dict = torch.load(path, map_location='cpu')
            self.model.load_state_dict(state_dict)

            for param in self.model.parameters():
                param.requires_grad = False

        self.avg_pool = AvgPool1d(kernel_size=16, stride=16)

    def forward(self, x):
        knowledges = {}
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.model.spatial(x)
        knowledges['res_fm'] = x
        knowledges['logits'] = self.fc(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.model.temporal(x).transpose(1, 2).contiguous()
        knowledges['temporal'] = self.avg_pool(x.clone().transpose(1, 2)).transpose(1, 2).squeeze()
        x = x.contiguous().view(num_batches * length, -1)
        x = self.model.regressor(x)
        x = x.view(num_batches, length, -1)
        return x, knowledges


class kd_res50_backup(kd_2d1d):
    def __init__(self, backbone_state_dict='', backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir='', folder='2d1d', knowledge=[],
                 role='student'):
        super().__init__(backbone_state_dict=backbone_state_dict, backbone_mode=backbone_mode, modality=modality,
                         embedding_dim=embedding_dim, channels=channels,
                         output_dim=output_dim, kernel_size=kernel_size, dropout=dropout, root_dir=root_dir, folder=folder,
                         knowledge=knowledge, role=role)

    def forward(self, x):
        knowledges = {}
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.model.spatial(x)
        knowledges['res_fm'] = x
        return x, knowledges


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from project.emotion_analysis_on_mahnob_hci.regression.knowledge_distillation_offline.configs import \
        config_knowledge_distillation as configs

    backbone_mode = configs['2d1d']['backbone_mode']
    modality = ['frame']
    embedding_dim = configs['2d1d']['cnn1d_embedding_dim']
    channels = configs['2d1d']['cnn1d_channels']
    kernel_size = configs['2d1d']['cnn1d_kernel_size']
    dropout = configs['2d1d']['cnn1d_dropout']
    model_load_path = "/home/zhangsu/phd2/load"

    teacher = kd_2d1d(backbone_state_dict='mahnob_reg_v_0', backbone_mode=backbone_mode, modality=modality,
                      embedding_dim=embedding_dim, channels=channels,
                      output_dim=1, kernel_size=kernel_size, dropout=dropout, root_dir=model_load_path, role="teacher")
    teacher.init(0)
    teacher.to(device)

    inputs = torch.zeros((2, 96, 3, 40, 40), dtype=torch.float32)
    inputs = inputs.to(device)
    outputs, knowledges = teacher(inputs)
    print(0)
