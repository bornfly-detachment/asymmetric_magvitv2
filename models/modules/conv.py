import logging
from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.nn import Module
from models.utils.util import (is_odd, cast_tuple)
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from collections import deque
logpy = logging.getLogger(__name__)

class CausalConv3dPlainAR(Module):
    def __init__(
            self,
            chan_in,
            chan_out,
            kernel_size: Union[int, Tuple[int, int, int]],
            pad_mode='constant',
            is_checkpoint=False,
            **kwargs
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)
        self.is_checkpoint = is_checkpoint
        self.pad_mode = pad_mode
        if isinstance(stride, int):
            stride = (stride, 1, 1)
        time_pad = dilation * (time_kernel_size - 1) + max((1 - stride[0]), 0)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.cache_front_feat = deque()

    # def forward(self, x):
    #     return checkpoint(self._forward, x)

    def forward(self, x, is_init_image=True):
        if self.is_checkpoint:
            return checkpoint(self._forward, x, is_init_image, use_reentrant=False)
        else:
            return self._forward(x, is_init_image)

    def _forward(self, x, is_init_image):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'
        if is_init_image:
            x = F.pad(x, self.time_causal_padding, mode=pad_mode)
            while len(self.cache_front_feat) > 0:
                self.cache_front_feat.pop()
                torch.cuda.empty_cache()
            self.cache_front_feat.append(x[:, :, -2:].clone().detach())
        else:
            x = F.pad(x, self.time_uncausal_padding, mode=pad_mode)
            video_front_context = self.cache_front_feat.pop()

            if video_front_context.shape[2] == 1:
                video_front_context = video_front_context.repeat(1, 1, 2, 1, 1)

            while len(self.cache_front_feat) > 0:
                self.cache_front_feat.pop()
                torch.cuda.empty_cache()

            x = torch.cat([video_front_context, x], dim=2)
            self.cache_front_feat.append(x[:, :, -2:].clone().detach())
            del video_front_context
            torch.cuda.empty_cache()
        x = self.conv(x)
        return x
