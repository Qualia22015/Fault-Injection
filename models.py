import torch
import torch.nn as nn
import torch.nn.functional as F


# 3-1. ResNet 기본 블록
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 3-2. 메인 모델 (FAT 생성용, 1024차원 압축)
class CNN_50_Layer(nn.Module):
    def __init__(self, block, num_blocks_list, num_classes=10):
        super(CNN_50_Layer, self).__init__()
        self.channels = [32, 64, 128, 256, 512]
        self.in_channels = self.channels[0]
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self._make_layer(block, self.channels[0], num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(block, self.channels[1], num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[2], num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[3], num_blocks_list[3], stride=1)
        self.layer5 = self._make_layer(block, self.channels[4], num_blocks_list[4], stride=1)
        self.linear = nn.Linear(self.channels[4], num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        fm0 = F.relu(self.bn1(self.conv1(x)))
        fm1 = self.layer1(fm0)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        fm5 = self.layer5(fm4)

        out = F.adaptive_avg_pool2d(fm5, (1, 1))
        out = out.view(out.size(0), -1)
        classification_out = self.linear(out)

        # FAT 생성 (압축)
        fm0_sum = torch.sum(fm0, dim=(2, 3))
        fm1_sum = torch.sum(fm1, dim=(2, 3))
        fm2_sum = torch.sum(fm2, dim=(2, 3))
        fm3_sum = torch.sum(fm3, dim=(2, 3))
        fm4_sum = torch.sum(fm4, dim=(2, 3))
        fm5_sum = torch.sum(fm5, dim=(2, 3))

        compressed_fat = torch.cat(
            [fm0_sum, fm1_sum, fm2_sum, fm3_sum, fm4_sum, fm5_sum], dim=1
        )

        return classification_out, compressed_fat
