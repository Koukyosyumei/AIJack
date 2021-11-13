import torch
import torch.nn as nn
import torch.nn.functional as F
from secure_ml.utils import Conv2d


class ResBlock(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, ks=3, bn=False, reduce=1):
        super(ResBlock, self).__init__()

        self.reduce = reduce
        self.bn = bn

        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=ks,
            stride=reduce,
        )
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.conv2 = Conv2d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=ks,
        )
        self.conv3 = Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=ks,
            stride=reduce,
        )

    def forward(self, inputs):

        x = inputs
        if self.bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        if self.bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        if self.reduce > 1:
            inputs = self.conv3(inputs)

        return inputs + x


class Resnet(nn.Module):
    def __init__(self, level=4):
        super(Resnet, self).__init__()

        self.level = level
        self.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResBlock(input_dim=64, output_dim=64)
        self.res2 = ResBlock(input_dim=64, output_dim=128, reduce=2)
        self.res3 = ResBlock(input_dim=128, output_dim=128)
        self.res4 = ResBlock(input_dim=128, output_dim=256, reduce=2)

    def forward(self, xin):
        x = self.conv1(xin)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.res1(x)

        if self.level == 1:
            return x
        x = self.res2(x)
        if self.level == 2:
            return x
        x = self.res3(x)
        if self.level == 3:
            return x
        x = self.res4(x)
        if self.level == 4:
            return x
        else:
            raise Exception("No level %d" % self.level)


class Pilot(nn.Module):
    def __init__(self, level=4):
        super(Pilot, self).__init__()

        self.level = level

        self.conv1 = Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=2
        )
        self.conv2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.conv3 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2
        )
        self.conv4 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1
        )
        self.conv5 = Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2
        )
        self.conv6 = Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1
        )

    def forward(self, xin):

        x = self.conv1(xin)
        x = nn.SiLU()(x)
        if self.level == 1:
            x = self.conv2(x)
            return x

        x = self.conv3(x)
        x = nn.SiLU()(x)
        if self.level <= 3:
            x = self.conv4(x)
            return x

        x = self.conv5(x)
        x = nn.SiLU()(x)
        if self.level <= 4:
            x = self.conv6(x)
            return x
        else:
            raise Exception("No level %d" % self.level)


class Decoder(nn.Module):
    def __init__(self, input_dim=256, level=4, channels=1):
        super(Decoder, self).__init__()

        self.level = level

        self.conv1 = nn.ConvTranspose2d(
            in_channels=input_dim, out_channels=256, kernel_size=3, padding=1, stride=2
        )
        self.conv2 = Conv2d(
            in_channels=256, out_channels=channels, kernel_size=3, stride=1
        )
        self.conv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=0, stride=2
        )
        self.conv4 = Conv2d(
            in_channels=128, out_channels=channels, kernel_size=3, stride=1
        )
        self.conv5 = nn.ConvTranspose2d(
            in_channels=128, out_channels=channels, kernel_size=2, padding=1, stride=2
        )

    def forward(self, xin):

        x = self.conv1(xin)
        x = F.relu(x)
        if self.level == 1:
            x = self.conv2(x)
            x = torch.tanh(x)
            return x

        x = self.conv3(x)
        x = F.relu(x)
        if self.level <= 3:
            x = self.conv4(x)
            x = torch.tanh(x)
            return x

        x = self.conv5(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, level=4, bn=False):
        super(Discriminator, self).__init__()

        self.level = level

        self.conv1 = Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2)
        self.conv2 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.res1 = ResBlock(input_dim=256, output_dim=256, bn=bn)
        self.res2 = ResBlock(input_dim=256, output_dim=256, bn=bn)
        self.res3 = ResBlock(input_dim=256, output_dim=256, bn=bn)
        self.res4 = ResBlock(input_dim=256, output_dim=256, bn=bn)
        self.res5 = ResBlock(input_dim=256, output_dim=256, bn=bn)
        self.res6 = ResBlock(input_dim=256, output_dim=256, bn=bn)
        self.conv5 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.fla = nn.Flatten()
        self.lin = nn.Linear(256, 1)

    def forward(self, xin):
        x = xin
        if self.level == 1:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
        if self.level <= 3:
            x = self.conv3(x)
        if self.level <= 4:
            x = self.conv4(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv5(x)
        x = self.fla(x)
        x = self.lin(x)
        # x = torch.sigmoid(x)

        return x
