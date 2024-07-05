
import logging
import torch
import torch.nn as nn
logpy = logging.getLogger(__name__)
from .conv import CausalConv3dPlainAR
from .ops import Normalize, nonlinearity

class Resnet3DBlock(nn.Module):
    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout,
            temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = CausalConv3dPlainAR(in_channels, out_channels, kernel_size=3)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3dPlainAR(out_channels, out_channels, kernel_size=3)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3dPlainAR(in_channels, out_channels, kernel_size=3)
            else:
                self.nin_shortcut = CausalConv3dPlainAR(in_channels, out_channels, kernel_size=1)

    def forward(self, x, temb=None, is_init_image=True):

        h = x.clone()
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h, is_init_image)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h, is_init_image)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, is_init_image)
            else:
                x = self.nin_shortcut(x, True)
        x = x + h
        return x