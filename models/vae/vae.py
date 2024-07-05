import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from einops import rearrange
from models.utils.ckpt_utils import load_checkpoint
logpy = logging.getLogger(__name__)
from models.utils.registry import MODELS, build_module

from transformers import PretrainedConfig, PreTrainedModel
from models.utils.util import to_torch_dtype
import os

class AsymmetricMagVitV2PiplineConfig(PretrainedConfig):
    model_type = "AsymmetricMagVitV2Pipline"

    def __init__(
        self,
        from_pretrained=None,
        **kwargs,
    ):
        super().__init__(**kwargs)


class AsymmetricMagVitV2Pipline(PreTrainedModel):
    config_class = AsymmetricMagVitV2PiplineConfig
    def __init__(self, config: AsymmetricMagVitV2PiplineConfig):
        super().__init__(config=config)
        print('AsymmetricMagVitV2Pipline config', config)
        self.encoder = build_module(config.encoder, MODELS)
        self.decoder = build_module(config.decoder, MODELS)
        self.regularizer = build_module(config.regularizer, MODELS)

    def set_encoder_frame(self, frame):
        self.encoder.video_frame_num = frame

    def set_decoder_frame(self, frame):
        self.decoder.video_frame_num = frame


    def get_input(self, batch: Dict) -> torch.Tensor:
        x = batch[self.input_key]
        if x.ndim == 5:
            batch_size = x.shape[0]
            num_video_frames = x.shape[2]
            batch["num_video_frames"] = int(batch["num_video_frames"][0])
            batch[self.input_key] = rearrange(batch[self.input_key], "b c t h w -> (b t) c h w")

            return batch[self.input_key], num_video_frames, batch_size
        num_video_frames = int(batch["num_video_frames"])
        batch_size = x.shape[0] / num_video_frames
        return batch[self.input_key], num_video_frames, batch_size

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + list(self.encoder.parameters())
        params = params + list(self.decoder.parameters())
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
            self,
            x: torch.Tensor,
            video_frame_num,
            is_init_image: bool = True,
            return_reg_log: bool = False,
            unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x, video_frame_num, is_init_image)
        if unregularized:
            if return_reg_log:
                return z, dict()
            else:
                return z
        z, reg_log = self.regularizer(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, batch_size, is_init_image: bool = True, **kwargs) -> torch.Tensor:
        x = self.decoder(z, batch_size, is_init_image, **kwargs)
        return x

    def forward(
            self, x: torch.Tensor, video_frame_num, batch_size, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, video_frame_num, return_reg_log=True)
        dec = self.decode(z, batch_size, **additional_decode_kwargs)
        return z, dec, reg_log

def Asymmetric_MagVitV2(
        from_pretrained=None,
        local_files_only=False,
        force_huggingface=False,
        dtype='bf16',
        z_channels=4
):
    encoder_config = dict(
        type="VideoEncoder",

        double_z=True,
        z_channels=z_channels,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        video_frame_num=17,
        down_sampling_layer=[1, 2]

    )

    decoder_config = dict(
        type="VideoDecoder",

        double_z=True,
        z_channels=z_channels,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        video_frame_num=17,
        temporal_up_layers=[2, 3]

    )

    regularizer_config = dict(
        type="DiagonalGaussianRegularizer",
    )

    kwargs = dict(
        encoder=encoder_config,
        decoder=decoder_config,
        regularizer=regularizer_config
    )

    dtype = to_torch_dtype(dtype)

    if force_huggingface or (from_pretrained is not None and not os.path.exists(from_pretrained)):
        model = AsymmetricMagVitV2Pipline.from_pretrained(from_pretrained, **kwargs)
    else:
        config = AsymmetricMagVitV2PiplineConfig(**kwargs)
        model = AsymmetricMagVitV2Pipline(config)

        if from_pretrained:
            load_checkpoint(model, from_pretrained)

    model.encoder.to(dtype)
    model.decoder.to(dtype)
    model.regularizer.to(dtype)
    return model

