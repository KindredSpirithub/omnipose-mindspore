# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

import math
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import initializer
BN_MOMENTUM=0.9

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 


class SepConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, has_bias=True, pad_mode='zeros', depth_multiplier=1):
        super(SepConv2d, self).__init__()

        intermediate_channels = in_channels * depth_multiplier

        self.spatialConv = nn.Conv2d(in_channels, intermediate_channels,kernel_size, stride,
             padding=padding, dilation=dilation, group=in_channels, pad_mode='pad', has_bias=has_bias)

        self.pointConv = nn.Conv2d(intermediate_channels, out_channels,
             kernel_size=1, stride=1, padding=0, dilation=1, has_bias=has_bias)

        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.spatialConv(x)
        x = self.relu(x)
        x = self.pointConv(x)

        return x

conv_dict = {
    'CONV2D': nn.Conv2d,
    'SEPARABLE': SepConv2d
}

class _AtrousModule(nn.Cell):
    def __init__(self, conv_type, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.conv = conv_dict[conv_type]
        self.atrous_conv = self.conv(inplanes, planes, kernel_size=kernel_size,
                            stride=1, pad_mode='pad', padding=padding, dilation=dilation, has_bias=False)

        self.bn = BatchNorm(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

        self._init_weights()

    def construct(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weights(self):
        print("=> init weights from normal distribution")
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                normal_init = mindspore.common.initializer.HeNormal()
                cell.weight.set_data(initializer(normal_init, cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(initializer(0, cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer(1, cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(0, cell.beta.shape, cell.beta.dtype))

class wasp(nn.Cell):
    def __init__(self, inplanes, planes, upDilations=[4,8]):
        super(wasp, self).__init__()
        dilations = [6, 12, 18, 24]
        BatchNorm = nn.BatchNorm2d

        self.aspp1 = _AtrousModule('CONV2D', inplanes, planes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _AtrousModule('CONV2D', planes, planes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _AtrousModule('CONV2D', planes, planes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _AtrousModule('CONV2D', planes, planes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = ops.AdaptiveAvgPool2D((1, 1))
        self.avg_CBR = nn.SequentialCell(
                                            #  nn.AdaptiveAvgPool2d((1, 1)),
                                            #  ops.AdaptiveAvgPool2D((1, 1)),
                                             nn.Conv2d(inplanes, planes, 1, stride=1, has_bias=False),
                                             nn.BatchNorm2d(planes, momentum=0.1),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(5*planes, planes, 1, has_bias=False)
        self.conv2 = nn.Conv2d(planes,planes,1, has_bias=False)
        self.bn1 = BatchNorm(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.cat = ops.Concat()
        self.resize_bilinear = nn.ResizeBilinear()
        self._init_weights()

    def construct(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        x5 = self.avg_CBR(x5)
        x5 = self.resize_bilinear(x5, size=x4.shape[2:])
        x = self.cat((x1, x2, x3, x4, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weights(self):
        print("=> init weights from normal distribution")
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                normal_init = mindspore.common.initializer.HeNormal()
                cell.weight.set_data(initializer(normal_init, cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(initializer(0, cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer(1, cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(0, cell.beta.shape, cell.beta.dtype))

    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


def build_wasp(inplanes, planes, upDilations):
    return wasp(inplanes, planes, upDilations)


class WASPv2(nn.Cell):
    def __init__(self, conv_type, inplanes, planes, n_classes=17):
        super(WASPv2, self).__init__()

        # WASP
        dilations = [1, 6, 12, 18]
        # dilations = [1, 12, 24, 36]
        
        # convs = conv_dict[conv_type]

        reduction = planes // 8

        BatchNorm = nn.BatchNorm2d

        self.aspp1 = _AtrousModule(conv_type, inplanes, planes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.relu = nn.ReLU()


        # self.global_avg_pool = ops.AdaptiveAvgPool2D((1, 1))
        self.global_avg_pool = nn.AvgPool2d(kernel_size=(96,72))
        self.CBR = nn.SequentialCell(
                                            # nn.AdaptiveAvgPool2d((1, 1)),
                                            #  ops.AdaptiveAvgPool2D((1, 1)),
                                             nn.Conv2d(inplanes, planes, 1, stride=1, has_bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU())
         
        self.conv1 = nn.Conv2d(5*planes, planes, 1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, reduction, 1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(reduction, momentum=BN_MOMENTUM)

        self.last_conv = nn.SequentialCell(nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(),
                                       nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(),
                                       nn.Conv2d(planes, n_classes, kernel_size=1, stride=1))
        self.cat = ops.Concat(axis=1)
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, x, low_level_features):
        input = x
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)
        x5 = self.global_avg_pool(x)
        x5 = self.CBR(x5)
        x5 = self.resize_bilinear(x5, size=x4.shape[2:])

        x = self.cat((x1, x2, x3, x4, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = self.cat((x, low_level_features))
        x = self.last_conv(x)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x
    
    

