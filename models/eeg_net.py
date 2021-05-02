import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def init_blocks(self, dropoutRate, *args, **kwargs):

        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            # DepthwiseConv2D =======================
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate))

        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate))

        return nn.Sequential(block1, block2)

    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Linear(inputSize, n_classes, bias=False)

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, num_classes=4, num_channels=60, num_samples=151,
                 dropout_rate=0.5, kernel_length=64, kernel_length2=16, F1=8,
                 D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = num_samples
        self.n_classes = num_classes
        self.channels = num_channels
        self.kernelLength = kernel_length
        self.kernelLength2 = kernel_length2
        self.dropoutRate = dropout_rate

        self.blocks = self.init_blocks(dropout_rate)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, num_channels, num_samples)
        self.logit = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.logit(x)

        return x

def categorical_cross_entropy(y_pred, y_true):
    # y_pred = y_pred.cuda()
    # y_true = y_true.cuda()
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()