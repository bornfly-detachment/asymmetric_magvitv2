
from models.modules.attention import AttnBlock
from models.modules.resblock import Resnet3DBlock
from models.modules.updownsample import Upsample2D, Upsample3D
from models.modules.ops import Normalize, nonlinearity
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from collections import deque
from einops import rearrange
import torch
from models.utils.registry import MODELS, build_module
from models.modules.conv import CausalConv3dPlainAR

@MODELS.register_module()
class VideoDecoder(nn.Module):
    def __init__(
            self,
            ch,
            out_ch,
            in_channels,
            z_channels,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            dropout=0.0,
            resamp_with_conv=True,
            give_pre_end=False,
            tanh_out=False,
            use_linear_attn=False,
            video_frame_num=1,
            temporal_up_layers=[2, 3],
            is_checkpoint=False,
            temporal_downsample=4,
            **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.video_frame_num = video_frame_num
        self.temporal_up_layers = temporal_up_layers
        self.is_checkpoint = is_checkpoint
        self.temporal_downsample = temporal_downsample


        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = CausalConv3dPlainAR(z_channels, block_in, kernel_size=3)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Resnet3DBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = Resnet3DBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up_id = len(self.temporal_up_layers)
        self.cur_video_frame_num = self.video_frame_num // 2 ** self.up_id + 1
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Resnet3DBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level in self.temporal_up_layers:
                    up.upsample = Upsample3D(block_in, resamp_with_conv)
                    self.cur_video_frame_num = self.cur_video_frame_num * 2
                else:
                    up.upsample = Upsample2D(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.time_causal_padding = (1, 1, 1, 1, 2, 0)
        self.time_uncausal_padding = (1, 1, 1, 1, 0, 0)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3)
        self.conv_out_cache_front_feat = deque()

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, batch_size, is_init_image, **kwargs):
        self.kwargs = kwargs
        if self.is_checkpoint:
            return checkpoint(self._forward, z, batch_size, is_init_image)
        else:
            return self._forward(z, batch_size, is_init_image)

    def _forward(self, z, batch_size, is_init_image):
        kwargs = self.kwargs
        b = batch_size
        z = rearrange(z, "(b t) c h w -> b c t h w", b=b)

        self.last_z_shape = z.shape
        temb = None

        h = self.conv_in(z, is_init_image=is_init_image)
        h = self.mid.block_1(h, temb, is_init_image=is_init_image, **kwargs)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, is_init_image=is_init_image, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, is_init_image=is_init_image, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, is_init_image=is_init_image, is_split=True)
        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        if is_init_image:
            while len(self.conv_out_cache_front_feat) > 0:
                self.conv_out_cache_front_feat.pop()
            h = F.pad(h, self.time_causal_padding, mode='constant')
            self.conv_out_cache_front_feat.append(h[:, :, -2:])
        else:

            h = F.pad(h, self.time_uncausal_padding, mode='constant')
            video_front_context = self.conv_out_cache_front_feat.pop()
            h = torch.cat([video_front_context, h], dim=2)

            while len(self.conv_out_cache_front_feat) > 0:
                self.conv_out_cache_front_feat.pop()

            self.conv_out_cache_front_feat.append(h[:, :, -2:])
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)

        if is_init_image:
            h = h[:, :, (self.temporal_downsample - 1):]
        h = rearrange(h, "b c t h w -> (b t) c h w ", b=b)
        return h
