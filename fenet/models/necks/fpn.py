import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..registry import NECKS


import pdb


@NECKS.register_module
class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention=False,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier',
                               layer='Conv2d',
                               distribution='uniform'),
                 cfg=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.attention = attention
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        # 判断现在是否已经是最后一层
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        # 设置 金字塔模型总共需要多少层
        # 从start的层数开始 到end的层数结束  每一层设定是否加额外的Relu 默认设置加
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 整个金字塔模型的构建 流程
        for i in range(self.start_level, self.backbone_end_level):
            # 横向卷积层 下采样输出+上采样的上一张图 = 本次卷积值
            # 先根据输入通道数量 和 输出通道数量设置 层内的卷积 1*1卷积核 得出每一个金字塔层的卷积结构
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            # 前向卷积层 下采样层
            # 卷积完后的结构作为输入 3*3卷积核 做一个同等尺度的卷积, 将其输出作为金字塔的其中一层feature map
            fpn_conv = ConvModule(out_channels,
                                  out_channels,
                                  3,
                                  padding=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg,
                                  inplace=False)
            # 将横向卷积的结构加入到层间卷积列表
            self.lateral_convs.append(l_conv)
            # 将前向卷积输出结构加入到金字塔列表,作为下采样的模型
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        # 在下采样结束之后额外增加不属于金字塔结构的下采样卷积层
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels,
                                            out_channels,
                                            3,
                                            stride=2,
                                            padding=1,
                                            conv_cfg=conv_cfg,
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) >= len(self.in_channels)

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # resnet
        # input 0 ([24, 128, 40, 100])
        # input 1 ([24, 256, 20, 50])
        # input 2 ([24, 512, 10, 25])
        

        # dla34 多了一个，会删除第一个input
        # input 0 torch.Size([24, 64, 80, 200])
        # input 1 torch.Size([24, 128, 40, 100])
        # input 2 torch.Size([24, 256, 20, 50])
        # input 2 torch.Size([24, 512, 10, 25])

        # build laterals
        # 创建上采样模型 横向卷积输出的值
        # 构造一个特征图列表，laterals通过对每个输入特征图应用横向卷积来调用。横向卷积存储在lateral_convs列表中
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # dla resnet一样
        #  layer1 [24,64,40,100] = [b,c,h,w]
        #  layer2 [24,64,20,50]
        #  layer3 [24,64,10,25]
        # pdb.set_trace()

        # build top-down path
        # 创建下采样的通道 根据给定下采样层数设置
        # 通过使用一组上采样操作对较高级别的特征图进行上采样以匹配较低级别特征图的分辨率来构建 FPN 的自顶向下路径。
        # 然后将上采样的特征图添加到较低级别的特征图以创建组合特征图
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 size=prev_shape,
                                                 **self.upsample_cfg)

        # pdb.set_trace()
        # build outputs
        # part 1: from original levels
        # 构建整体输出
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # fpn layer1 [24,64,40,100] = [b,c,h,w]
        # fpn layer2 [24,64,20,50]
        # fpn layer3 [24,64,10,25]
        # pdb.set_trace()
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # pdb.set_trace()
        # outs layer1 [24,64,40,100] = [b,c,h,w]
        # outs layer2 [24,64,20,50]
        # outs layer3 [24,64,10,25]

        return tuple(outs)
