from .conv import CausalConv3dPlainAR
import torch.nn as nn
from .ops import Normalize
import torch
from einops import rearrange

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = CausalConv3dPlainAR(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = CausalConv3dPlainAR(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = CausalConv3dPlainAR(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = CausalConv3dPlainAR(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def attention(self, h_, is_init_image=True) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_, is_init_image)
        k = self.k(h_, is_init_image)
        v = self.v(h_, is_init_image)
        # print('h_', h_.shape)

        b, c, t, h, w = q.shape
        q, k, v = map(
            lambda x: rearrange(x, "b c t h w -> b 1 (t h w) c").contiguous(), (q, k, v)
        )
        h_ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        return rearrange(h_, "b 1 (t h w) c -> b c t h w", t=t, h=h, w=w, c=c, b=b)

    def forward(self, x):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_
