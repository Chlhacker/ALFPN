import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from DCNv2.dcn_v2 import DCNv2 as dcn_v2

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import ipdb

from ..builder import NECKS
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import fvcore.nn.weight_init as weight_init
from mmdet.models.Deformconv import DeformConv2d
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

		# 如果下面这个原论文代码用不了的话，可以换成另一个试试
        out = identity * a_w * a_h
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class feature_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 分组卷积+大卷积核
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 在1x1之前使用唯一一次LN做归一化
        self.norm = LayerNorm(dim, eps=1e-6)
        # 全连接层跟1x1conv等价，但pytorch计算上fc略快
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # 整个block只使用唯一一次激活层
        self.act = nn.GELU()
        # 反瓶颈结构，中间层升维了4倍
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # gamma的作用是用于做layer scale训练策略
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # drop_path是用于stoch. depth训练策略
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # 由于用FC来做1x1conv，所以需要调换通道顺序
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# class DC_FeatureAlign(nn.Module):  # FaPN full version
#     def __init__(self, in_nc=128, out_nc=128):
#         super().__init__()
#         self.feature_offset = nn.Conv2d(out_nc , out_nc, kernel_size=1, stride=1, padding=0, bias=False)
#         self.dcn = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
#         self.relu = nn.ReLU(inplace=True)
#         self.DC = DeformConv2d(in_nc, out_nc, modulation=True)
#         self.redu_ch = nn.Conv2d(in_nc, in_nc//2 ,kernel_size=1)
#         self.norm = nn.BatchNorm2d(out_nc)
#         weight_init.c2_xavier_fill(self.feature_offset)
#
#     def forward(self, feature_bottom, feature_top, main_path=None):
#         #ipdb.set_trace()
#         # feat_arm = self.lateral_conv(feat_l)  #0~1 * feats 经过 FDM 模块
#         #offset = self.feature_offset(torch.cat([feature_bottom, feature_top * 2], dim=1))  # concat for offset by compute the dif
#
#         cat_feature = torch.cat((self.redu_ch(feature_bottom), feature_top), dim=1)
#
#
#
#         feature_Align = self.relu(self.norm(self.DC(cat_feature)))  # [feat, offset]
#         return feature_Align + feature_bottom

class DC_FeatureAlign(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128):
        super().__init__()
        self.feature_offset = nn.Conv2d(out_nc , out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.dcn = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
        self.relu = nn.ReLU(inplace=True)
        self.DC = DeformConv2d(in_nc, out_nc, modulation=True)
        self.redu_ch = nn.Conv2d(in_nc, in_nc//2 ,kernel_size=1)
        self.exp_ch = nn.Conv2d(in_nc//2, in_nc, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_nc)
        self.AFM = AFM_Lite(in_channel= in_nc)
        weight_init.c2_xavier_fill(self.feature_offset)

    def forward(self, feature_bottom, feature_top, main_path=None):

        #ipdb.set_trace()
        #caca = self.exp_ch(feature_top)
        # feat_arm = self.lateral_conv(feat_l)  #0~1 * feats 经过 FDM 模块
        #offset = self.feature_offset(torch.cat([feature_bottom, feature_top * 2], dim=1))  # concat for offset by compute the dif

        #cat_feature = torch.cat((self.redu_ch(feature_bottom), feature_top), dim=1)
        cat_feature = self.AFM(feature_bottom, self.exp_ch(feature_top))


        feature_Align = self.relu(self.norm(self.DC(cat_feature)))  # [feat, offset]
        return feature_Align + feature_bottom


class AFM_Lite(nn.Module):
    def __init__(self, in_channel, rfb=False, vis=False):
        super(AFM_Lite, self).__init__()

        compress_c = 8 if rfb else 16
        self.weight_level_0 = nn.Conv2d(in_channel, compress_c, 1, 1)
        self.weight_level_1 = nn.Conv2d(in_channel, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, level_0, level_1):
        #ipdb.set_trace()

        h, w = level_0.shape[2], level_0.shape[3]
        level_0_resized = level_0
        level_1_resized = level_1




        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)

        levels_weights_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weights = self.weight_levels(levels_weights_v)
        levels_weights = F.softmax(levels_weights, dim=1)

        out = level_0_resized * levels_weights[:, 0:1, :, :] + \
              level_1_resized * levels_weights[:, 1:, :, :]
        return out


@NECKS.register_module()
class ALFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ALFPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ALFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.sub_pixshuffle = nn.PixelShuffle(2)
        self.upsample_cfg = upsample_cfg.copy()


        # self.P3_channel_change = nn.Conv2d(128, in_channels[0], kernel_size=1)
        # self.P2_conv = nn.Conv2d(in_channels[0], in_channels[0]//2, kernel_size=1)
        # # self.attention_c = CoordAtt(inp=256, oup=256)
        # self.attention_c = ChannelAttention(in_planes=256)
        # self.attention_P3 = CoordAtt(inp=in_channels[1], oup=in_channels[1])
        #
        # self.P3_conv = nn.Conv2d(in_channels[1], in_channels[1]//2, kernel_size=1)
        # # self.attention_d = CoordAtt(inp=512, oup=512)
        # self.attention_d = ChannelAttention(in_planes=512)
        # self.attention_P4 = CoordAtt(inp=in_channels[2],oup=in_channels[2])
        # self.P4_channel_change = nn.Conv2d(256, in_channels[1], kernel_size=1)


        #############################################
        #P2
        self.P2_DC = DC_FeatureAlign(in_nc=in_channels[0], out_nc=in_channels[0])
        #self.P2_DC = DeformConv2d(in_channels[0], in_channels[0], modulation=True)
        self.P2_conv = nn.Conv2d(in_channels[0], in_channels[0] // 2, kernel_size=1)
        self.attention_P2 = ChannelAttention(in_planes=256)


        #############################################
        #P3
        self.P3_DC = DC_FeatureAlign(in_nc=in_channels[1], out_nc=in_channels[1])
        #self.P3_DC = DeformConv2d(in_channels[1], in_channels[1], modulation=True)
        self.P3_conv = nn.Conv2d(in_channels[1], in_channels[1] // 2, kernel_size=1)
        self.attention_P3 = ChannelAttention(in_planes=512)



        self.P3_up_module = nn.Sequential(
            feature_Block(dim=in_channels[1]),
            nn.PixelShuffle(2),
        )
        #############################################
        #P4
        self.P4_up_module = nn.Sequential(
            feature_Block(dim=in_channels[2]),
            nn.PixelShuffle(2)
        )


        #############################################
        #AFM



        #############################################





        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'


        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        #ipdb.set_trace()
        assert len(inputs) == len(self.in_channels)

        # build laterals
        ##############################################

        #P2 = torch.cat((self.P3_up_module(inputs[1]), self.P2_conv(inputs[0])), dim=1)
        # P2_final = P2*self.attention_P2(P2)
        #P3 = torch.cat((self.P4_up_module(inputs[2]), self.P3_conv(inputs[1])), dim=1)
        # P3_final = P3*self.attention_P3(p3)
        ##############################################
        #ipdb.set_trace()
        P2 = self.P2_DC(inputs[0], self.P3_up_module(inputs[1]))

        P2_final = P2 * self.attention_P2(P2)

        P3 = self.P3_DC(inputs[1], self.P4_up_module(inputs[2]))
        P3_final = P3 * self.attention_P3(P3)

        # P2_final = inputs[0] * self.attention_P2(
        #     torch.cat((self.P3_up_module(inputs[1]), self.P2_conv(inputs[0])), dim=1))
        # P3_final = inputs[1] * self.attention_P3(
        #     torch.cat((self.P4_up_module(inputs[2]), self.P3_conv(inputs[1])), dim=1))



        new_inputs = [P2_final, P3_final, inputs[2], inputs[3]]

        laterals = [
            lateral_conv(new_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]



        # build top-down path
        #ipdb.set_trace()
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 2, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        # laterals[0] = self.P2_cbam(laterals[0])
        # laterals[1] = self.P3_cbam(laterals[1])
        # laterals[2] = self.P4_cbam(laterals[2])
        # laterals[3] = self.P5_cbam(laterals[3])

        #ipdb.set_trace()
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        #ipdb.set_trace()



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
        return tuple(outs)
