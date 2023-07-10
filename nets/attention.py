import math

import torch
from torch import nn


class SEBlock(nn.Module):
    """
    SE: Squeeze-and-Excitation
    """

    def __init__(self, channel, ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """
    ECA: Efficient Channel Attention
    """

    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.conv(y.view(b, 1, c))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelBlock(nn.Module):
    """
    Channel Attention Module
    """

    def __init__(self, channel, ratio=16):
        super(ChannelBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_x = self.avg_pool(x).view(b, c)
        max_x = self.max_pool(x).view(b, c)

        avg_x = self.fc(avg_x).view(b, c, 1, 1)
        max_x = self.fc(max_x).view(b, c, 1, 1)

        out = avg_x + max_x
        out = self.sigmoid(out)
        return x * out.expand_as(x)


class SpatialBlock(nn.Module):
    """
    Spatial Attention Module
    """

    def __init__(self, kernel_size=7):
        super(SpatialBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        avg_x = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_x, avg_x], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out.expand_as(x)


class CBAMBlock(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    """

    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelBlock(channel, ratio)
        self.spatial_attention = SpatialBlock(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CABlock(nn.Module):
    """
    CA: Coordinate Attention
    """

    def __init__(self, channel, reduction=16):
        super(CABlock, self).__init__()

        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))

        middle_channel = max(channel // reduction, 8)

        self.conv_x = nn.Conv2d(in_channels=channel,
                                out_channels=middle_channel,
                                kernel_size=1,
                                stride=1,
                                bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(middle_channel)

        self.F_h = nn.Conv2d(in_channels=middle_channel,
                             out_channels=channel,
                             kernel_size=1,
                             stride=1,
                             bias=False)

        self.F_w = nn.Conv2d(in_channels=middle_channel,
                             out_channels=channel,
                             kernel_size=1,
                             stride=1,
                             bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        b, c, h, w = x.size()
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_x(torch.cat([x_h, x_w], dim=3)))

        x_cat_conv_split_h, x_cat_conv_split_w = torch.split(x_cat_conv_relu, [h, w], dim=3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h).permute(0, 1, 3, 2))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


if __name__ == '__main__':
    b, c, h, w = 1, 24, 224, 224
    input_x = torch.randn(b, c, h, w)

    se_block = SEBlock(c, 16)
    eca_block = ECABlock(c)
    cbam_block = CBAMBlock(c)
    ca_block = CABlock(c)

    se_out = se_block(input_x)
    eca_out = eca_block(input_x)
    cbam_out = cbam_block(input_x)
    ca_out = ca_block(input_x)

    print(se_out.shape)
    print(f'se_out:{se_out}')

    print(eca_out.shape)
    print(f'eca_out:{eca_out}')

    print(cbam_out.shape)
    print(f'cbam_out:{cbam_out}')

    print(ca_out.shape)
    print(f'ca_out:{ca_out}')

