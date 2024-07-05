
import logging
import torch
import torch.nn as nn
from einops import rearrange
logpy = logging.getLogger(__name__)
from .conv import CausalConv3dPlainAR

class Downsample2D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x, is_init_image=False):
        _, _, t, _, _ = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w", t=t)
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        x = rearrange(x, "(b t) c h w  -> b c t h w", t=t)
        return x

class Downsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = CausalConv3dPlainAR(in_channels, in_channels, kernel_size=3, stride=stride)

    def forward(self, x, is_init_image=True):
        if self.with_conv:
            x = self.conv(x, is_init_image)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class Res3DBlockUpsample(nn.Module):
    def __init__(self,
                 input_filters,
                 num_filters,
                 base_filters,
                 down_sampling_stride,
                 down_sampling=False,
                 down_sampling_temporal=None,
                 is_real_3d=True,
                 with_norm=True):
        super(Res3DBlockUpsample, self).__init__()
        self.num_filters = num_filters
        self.base_filters = base_filters
        self.input_filters = input_filters
        self.with_norm = with_norm
        if down_sampling:
            if is_real_3d and down_sampling_temporal:
                self.down_sampling_stride = down_sampling_stride
            else:
                self.down_sampling_stride = down_sampling_stride
        else:
            self.down_sampling_stride = [1, 1, 1]

        self.down_sampling = down_sampling

        self.act = nn.SiLU()
        self.conv1 = CausalConv3dPlainAR(num_filters, num_filters, kernel_size=[3, 3, 3], stride=[1, 1, 1])
        if self.with_norm:
            self.norm1 = nn.GroupNorm(32, num_filters)
        self.conv2 = CausalConv3dPlainAR(num_filters, num_filters, kernel_size=[3, 3, 3], stride=[1, 1, 1])
        if self.with_norm:
            self.norm2 = nn.GroupNorm(32, num_filters)
        if num_filters != input_filters or down_sampling:
            self.conv3 = CausalConv3dPlainAR(input_filters, num_filters, kernel_size=[1, 1, 1],
                                             stride=self.down_sampling_stride)
            if self.with_norm:
                self.norm3 = nn.GroupNorm(32, num_filters)

    def forward(self, x, is_init_image=False):
        identity = x
        out = self.conv1(x, is_init_image)
        if self.with_norm:
            out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out, is_init_image)
        if self.with_norm:
            out = self.norm2(out)
        if self.down_sampling or self.num_filters != self.input_filters:
            identity = self.conv3(identity, is_init_image)
            if self.with_norm:
                identity = self.norm3(identity)
        out += identity
        out = self.act(out)
        return out


class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, scale_factor=2):
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor

        self.conv3d = Res3DBlockUpsample(input_filters=in_channels,
                                         num_filters=in_channels,
                                         base_filters=in_channels,
                                         down_sampling_stride=(1, 1, 1),
                                         down_sampling=False)

    def _split_by_channel(self, x, split_size):
        slices = torch.split(x, split_size, dim=1)
        return slices

    def _split_by_batch(self, x, split_size):
        slices = torch.split(x, split_size, dim=0)
        return slices

    def forward(self, x, is_init_image=True, is_split=False):
        b, c, t, h, w = x.shape
        if is_split:
            split_size = c // 8
            x_slices = self._split_by_channel(x, split_size)

            x = [torch.nn.functional.interpolate(x, (
                x.shape[2] * self.scale_factor,
                x.shape[3] * self.scale_factor,
                x.shape[4] * self.scale_factor,
            ), mode='nearest') for x in x_slices]
            x = torch.cat(x, dim=1)
            identity = x
            if b > 2 and b % 2 == 0:
                split_size = b // 2
                x_slices = self._split_by_batch(x, split_size)
                x = [self.conv3d(b_x, is_init_image) for b_x in x_slices]
                x = torch.cat(x, dim=0)
            else:
                x = self.conv3d(x, is_init_image)

            return x + identity
        else:
            x = torch.nn.functional.interpolate(x, (
                x.shape[2] * self.scale_factor,
                x.shape[3] * self.scale_factor,
                x.shape[4] * self.scale_factor,
            ), mode='nearest')
            identity = x
            x = self.conv3d(x, is_init_image)
            return x + identity


class Upsample2D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x, is_init_image=True, is_split=False):
        b, _, _, _, _ = x.shape
        x = rearrange(x, "b c t h w->(b t) c h w", b=b)

        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        return x
