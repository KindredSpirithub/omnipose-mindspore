# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import math
import mindspore
import mindspore.nn as nn

from src.wasp import WASPv2
from mindspore import ops
from mindspore.nn import BatchNorm2d
from mindspore.common.initializer import initializer
BN_MOMENTUM = 0.9
logger = logging.getLogger(__name__)

class NoneCell(nn.Cell):
    """Cell doing nothing."""
    def __init__(self):
        super(NoneCell, self).__init__()
        self.name = "NoneCell"

    def construct(self, x):
        """NoneCell construction."""
        return x

class SepConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, bias=True, padding_mode='zeros', depth_multiplier=1):
        super(SepConv2d, self).__init__()

        intermediate_channels = in_channels * depth_multiplier

        self.spatialConv = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size, stride=stride, padding=padding, 
            dilation=dilation, group=in_channels, pad_mode="pad", has_bias=bias)

        self.pointConv = nn.Conv2d(intermediate_channels, out_channels,
             kernel_size=1, stride=1, padding=0, dilation=1, has_bias=bias)

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

# Change to the desired type of convolution
convs = conv_dict['CONV2D']
# convs = conv_dict['SEPARABLE']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return convs(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, pad_mode="pad", has_bias=False)
    # return convs(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Decoder(nn.Cell):
    def __init__(self, low_level_inplanes, planes, num_classes):
        super(Decoder, self).__init__()
        reduction = planes // 8

        self.conv1 = nn.Conv2d(low_level_inplanes, reduction, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(reduction, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.last_conv = nn.SequentialCell(nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=False),
                                       nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=False),
                                       nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                                       nn.ReLU(),
                                       nn.Dropout(0.9),
                                       nn.Conv2d(planes, num_classes, kernel_size=1, stride=1))
        self._init_weight()
        self.resize_bilinear = nn.ResizeBilinear()
        self.concat = ops.Concat(1)
        
    def construct(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = self.resize_bilinear(x, size=low_level_feat.size()[2:])
        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = self.concat((x, low_level_feat)) #[8,2323,192]
        x = self.last_conv(x)

        return x

    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, convs):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


class BasicBlock(nn.Cell):
    """BasicBlock definition."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.add = ops.Add()

    def construct(self, x):
        """BasicBlock construction."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu2(out)

        return out


class Bottleneck(nn.Cell):
    """Bottleneck definition."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad',
                               padding=1, has_bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               has_bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.add = ops.Add()

    def construct(self, x):
        """Bottleneck construction."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu3(out)

        return out


class HighResolutionModule(nn.Cell):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()
        ###add
        self.add = ops.Add()
        self.resize_bilinear = nn.ResizeBilinear()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialCell(
                convs(self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM),)

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.SequentialCell(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.CellList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i and (j-i) == 1:
                    fuse_layer.append(nn.SequentialCell(
                        convs(num_inchannels[j],
                                  num_inchannels[i],
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Conv2dTranspose(in_channels=num_inchannels[i],
                                           out_channels=num_inchannels[i],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           pad_mode="pad", 
                                           has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(),
                        self.gaussian_filter(num_inchannels[i], 3, 3)))
                elif j > i and (j-i) == 2:
                    fuse_layer.append(nn.SequentialCell(
                        convs(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  padding=0,
                                  has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Conv2dTranspose(in_channels=num_inchannels[i],
                                           out_channels=num_inchannels[i],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           pad_mode="pad", 
                                           has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(),
                        nn.Conv2dTranspose(in_channels=num_inchannels[i],
                                           out_channels=num_inchannels[i],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           pad_mode="pad", 
                                           has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(),
                        self.gaussian_filter(num_inchannels[i], 3, 3)))
                elif j > i and (j-i) == 3:
                    fuse_layer.append(nn.SequentialCell(
                        convs(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  padding=0,
                                  has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Conv2dTranspose(in_channels=num_inchannels[i],
                                           out_channels=num_inchannels[i],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           pad_mode="pad", 
                                           has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(),
                        nn.Conv2dTranspose(in_channels=num_inchannels[i],
                                           out_channels=num_inchannels[i],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           pad_mode="pad", 
                                           has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(),
                        nn.Conv2dTranspose(in_channels=num_inchannels[i],
                                           out_channels=num_inchannels[i],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           pad_mode="pad", 
                                           has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(),
                        self.gaussian_filter(num_inchannels[i], 3, 3)))
                elif j == i:
                    fuse_layer.append(NoneCell())
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.SequentialCell(
                                    convs(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, padding=1, pad_mode="pad", has_bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.SequentialCell(
                                    convs(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, padding=1, pad_mode="pad", has_bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                    nn.ReLU()))
                    fuse_layer.append(nn.SequentialCell(conv3x3s))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

        
    def gaussian_filter(self, channels, kernel_size, sigma):
        x_cord = mindspore.numpy.arange(kernel_size)
        # x_cord = torch.arange(kernel_size)
        x_grid = mindspore.numpy.tile(x_cord, kernel_size).view(kernel_size, kernel_size)
        # x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.transpose()
        stack = ops.Stack(axis=-1)
        xy_grid = stack([x_grid, y_grid])
        # cast = ops.Cast()
        # xy_grid = cast(xy_grid, mindspore.float32)
        # xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2
        sum = ops.ReduceSum()
        exp = ops.Exp()
        gaussian_kernel = (1./(2.*math.pi*sigma**2)) * exp(-sum((xy_grid - mean)**2., axis=-1) / (2*sigma**2))
        gaussian_kernel = gaussian_kernel / sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = mindspore.numpy.tile(gaussian_kernel, (channels, 1, 1, 1))

        ### 高斯核,分组卷积
        gaussian_fltr = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=kernel_size, padding=int(kernel_size//2), group=channels, pad_mode="pad", has_bias=False)

        gaussian_fltr.weight.set_data(gaussian_kernel)
        gaussian_fltr.weight.requires_grad = False

        return gaussian_fltr

    def get_num_inchannels(self):
        return self.num_inchannels

    def construct(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    # if i>j: ############remove upsample
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class OmniPose(nn.Cell):
    def __init__(self, cfg):
        self.inplanes = 64
        # extra = cfg.MODEL.EXTRA
        super(OmniPose, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1, self.flag1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2, self.flag2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3, self.flag3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)

        # If using WASPv1
        # self.wasp = build_wasp(48, 48,[0,0])
        # self.decoder = Decoder(256, 48, cfg.MODEL.NUM_JOINTS)

        # If using WASPv2
        # self.waspv2 = WASPv2('SEPARABLE', 48, 48, cfg.MODEL.NUM_JOINTS)
        self.waspv2 = WASPv2('SEPARABLE', 720, 192, cfg.MODEL.NUM_JOINTS)
        self.resize_bilinear = nn.ResizeBilinear()
        self.concat = ops.Concat(axis=1)
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        self.init_weights()


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        flag = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialCell(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, padding=1, pad_mode="pad", has_bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU()
                        )
                    )
                    flag.append("ops")
                else:
                    transition_layers.append(NoneCell())
                    flag.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    # 下采样，分辨率减半，通道加倍
                    conv3x3s.append(nn.Conv2d(
                                inchannels, outchannels, 3, 2, padding=1, pad_mode="pad", has_bias=False))
                    conv3x3s.append(nn.BatchNorm2d(outchannels, momentum=0.1))
                    conv3x3s.append(nn.ReLU())
                
                transition_layers.append(nn.SequentialCell(*conv3x3s))
                flag.append("ops")

        return nn.CellList(transition_layers), flag

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, pad_mode="pad", has_bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        i=1
        while i < blocks:
            layers.append(block(self.inplanes, planes))
            i+=1

        return nn.SequentialCell(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.SequentialCell(modules), num_inchannels

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1(x)

        low_level_feat = x

        x_list = []

        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.flag1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)

        # y_list = x_list
        # level_2 = y_list[0]
        
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.flag2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list) 
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.flag3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
                
        y_list = self.stage4(x_list)
        out1, out2, out3, out4 = y_list
        h, w = ops.Shape()(out1)[2:]
        x1 = out1
        # x1 = ops.Cast()(out1, mindspore.dtype.float32)
        x2 = self.resize_bilinear(out2, size=(h, w))
        x3 = self.resize_bilinear(out3, size=(h, w))
        x4 = self.resize_bilinear(out4, size=(h, w))

        x = self.concat((x1, x2, x3, x4))

        x = self.waspv2(x, low_level_feat)
        return x

    def init_weights(self, pretrained=''):
        print("=> init weights from normal distribution")
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                normal_init = mindspore.common.initializer.Normal(sigma=0.001)
                cell.weight.set_data(initializer(normal_init, cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(initializer(0, cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer(1, cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(0, cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2dTranspose):
                normal_init = mindspore.common.initializer.Normal(sigma=0.001)
                cell.weight.set_data(initializer(normal_init, cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(initializer(0, cell.bias.shape, cell.bias.dtype))

def get_omnipose(cfg, is_train, **kwargs):
    model = OmniPose(cfg, **kwargs)

    return model

