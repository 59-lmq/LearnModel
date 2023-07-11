"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""


from torchsummary import summary
from attention import *

import math

__all__ = ['mobilenet_v3_large', 'mobilenet_v3_small']

attention_list = [SEBlock, ECABlock, CBAMBlock, CABlock]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo. \n
    It ensures that all layers have a channel number that is divisible by 8 \n
    确保每层的通道数都能被8整除 \n
    e.p.:
        ①v=3, divisor=8 => return new_channel=8 \n
        ② v=16, divisor=8 => return new_channel=16 \n
        ③ v= 127, divisor=8 => return new_channel=127 \n
        It can be seen here: \n
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v: 输入的通道数
    :param divisor: 需要整除的数
    :param min_value: 最小的通道数，默认与需要整除的数（divisor）相同
    :return: v上下能被divisor整除的最大值
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, (int(v + divisor / 2) // divisor) * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < (0.9 * v):
        new_v += divisor
    return new_v


def make_divisible(x, divisible_by=8):
    """
    同上操作
    :param x: 输入的通道数
    :param divisible_by: 需要被整除的数
    :return: 返回上下能被divisible_by整除的最大值
    """
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        # ①卷积
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        # ②归一化
        nn.BatchNorm2d(oup),
        # ③激活
        HSwish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        HSwish()
    )


class InvertedResidual(nn.Module):
    """
    MobileNetV2的具有线性瓶颈的倒残差结构，也是MobileNetV3的基本模块，加入了SE注意力机制
    """
    def __init__(self, inp, hidden_dim, oup, kernel_size,
                 stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]  # 限定步长在 1 和 2
        assert kernel_size in [3, 5]  # 限定卷积核大小在 3 和 5

        self.identity = (stride == 1 and inp == oup)  # 在步长为1 且输入通道等于输出通道时选择shortcut
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw DepthWise Convolution 深度可分离卷积
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # 注意力机制
                attention_list[use_hs](hidden_dim, use_hs=use_hs) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw:也有使用 PW 做升维的，在 MobileNet v2 中就使用 PW 将 3 个特征图变成 6 个特征图，丰富输入数据的特征。
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=(kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # 注意力机制
                attention_list[use_hs](hidden_dim, use_hs=use_hs) if use_se else nn.Identity(),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear: Pointwise Convolution，俗称叫做 1x1 卷积，简写为 PW，主要用于数据降维，减少参数量。
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.identity:  # 需要skip connection（跳接）
            out = out+x
        return out


class MobileNetV3(nn.Module):
    """MobileNetV3 网络结构"""
    def __init__(self, cfgs, mode, num_classes=10, width_mult=1.):
        """

        :param cfgs: configs
        :param mode: 模式，large or small
        :param num_classes: 分类的数量
        :param width_mult: 宽度乘数，用于控制网络的宽度
        """
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer, 构建第一层，没有扩展expand
        # input_channel = _make_divisible(16 * width_mult, 8)
        input_channel = make_divisible(16 * width_mult, divisible_by=8)
        layers = [conv_3x3_bn(3, input_channel, 2)]  # [3, 224, 224] -> [16, 224, 224]
        # building inverted residual blocks，搭建 bottleneck residual blocks
        block = InvertedResidual  # 两头窄，中间胖的反残差结构，称为Inverted Residual
        exp_size = make_divisible(input_channel * 1, divisible_by=8)
        # k：kernel, t: expand_multi, c: output_channel, use_se:bool, use_hs:bool, s:stride
        for k, t, c, use_se, use_hs, s in self.cfgs:
            # output_channel = _make_divisible(c * width_mult, 8)
            # exp_size = _make_divisible(input_channel * t, 8)
            output_channel = make_divisible(c * width_mult, divisible_by=8)
            exp_size = make_divisible(input_channel * t, divisible_by=8)
            print('inp:{0}, exp:{2}, oup:{1}'.format(input_channel, output_channel, exp_size))
            layers.append(block(input_channel, exp_size, output_channel,
                                k, s, use_se, use_hs))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)
        # building last several layers，搭建最后几层
        self.conv = conv_1x1_bn(input_channel, exp_size)  # 1x1卷积升维
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 池化提取特征
        output_channel = {'large': 1280, 'small': 1024}
        if width_mult > 1.0:
            # output_channel = _make_divisible(output_channel[mode] * width_mult, 8)
            output_channel = make_divisible(output_channel[mode] * width_mult, divisible_by=8)
        else:
            output_channel = output_channel[mode]
        # output_channel = _make_divisible(output_channel[mode] * width_mult, 8)
        # if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()  # 初始化权重

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def mobilenet_v3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # kernel, exp_size, output_channel, SE, HS, stride
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]

    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenet_v3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # kernel, exp_size, output_channel, SE, HS, stride
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


def __test_function():
    # net_large = mobilenet_v3_large(num_classes=10)
    net_small = mobilenet_v3_small(num_classes=5)
    # print(net_large)
    summary(net_small.cuda(), (3, 224, 224))

    # print('Large Total params: %.2fM' % (sum(p.numel() for p in net_large.parameters())/1000000.0))
    print('Small Total params: %.2fM' % (sum(p.numel() for p in net_small.parameters()) / 1000000.0))
    # net_small = mobilenet_v3_small()

    target = torch.randn(3, 5).cpu()
    loss = nn.CrossEntropyLoss()
    m = nn.Softmax(dim=1)
    a = m(target)
    print(a)


if __name__ == '__main__':
    __test_function()


