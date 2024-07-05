
from models.modules.conv import CausalConv3dPlainAR
from models.modules.attention import AttnBlock
from models.modules.resblock import Resnet3DBlock
from models.modules.updownsample import Downsample2D, Downsample3D
from models.modules.ops import Normalize, nonlinearity
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from models.utils.util import is_odd
from models.utils.registry import MODELS, build_module

@MODELS.register_module()
class VideoEncoder(nn.Module):
    def __init__(
            self,
            *,
            ch,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks,
            dropout=0.0,
            resamp_with_conv=True,
            in_channels,
            z_channels,
            double_z=True,
            use_linear_attn=False,
            video_frame_num=1,
            down_sampling_layer=[1, 2],
            down_sampling=True,
            is_checkpoint=False,
            **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.video_frame_num = video_frame_num
        self.down_sampling_layer = down_sampling_layer
        self.down_sampling = down_sampling
        # self.temporal_length = max(self.video_frame_num // (2 ** len(self.down_sampling_layer)) + 1, 1)
        self.is_checkpoint = is_checkpoint

        # downsampling
        self.conv_in = CausalConv3dPlainAR(in_channels, self.ch, kernel_size=3)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Resnet3DBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level in self.down_sampling_layer:
                    down.downsample = Downsample3D(block_in, resamp_with_conv, stride=(2, 2, 2))
                else:
                    down.downsample = Downsample2D(block_in, resamp_with_conv)
            self.down.append(down)

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
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CausalConv3dPlainAR(block_in,
                                            2 * z_channels if double_z else z_channels,
                                            kernel_size=3)

    def forward(self, x, video_frame_num, is_init_image=True):
        if self.is_checkpoint:
            return checkpoint(self._forward, x, video_frame_num, is_init_image)
        else:
            return self._forward(x, video_frame_num, is_init_image)

    def _forward(self, x, video_frame_num, is_init_image):
        # timestep embedding
        temb = None
        t = video_frame_num
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        if is_odd(t):
            temporal_length = max(t // (2 ** len(self.down_sampling_layer)) + 1, 1)
        else:
            temporal_length = t // (2 ** len(self.down_sampling_layer))
        # downsampling
        hs = [self.conv_in(x, is_init_image)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, is_init_image)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1], is_init_image))
        h = hs[-1]
        h = self.mid.block_1(h, temb, is_init_image)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, is_init_image)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, is_init_image)

        if h.shape[2] != temporal_length:
            print("shape strange: ", x.shape)
        h = rearrange(h, "b c t h w  -> (b t) c h w", t=temporal_length)
        return h
