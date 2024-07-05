from abc import abstractmethod
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import \
    DiagonalGaussianDistribution
from .base import AbstractRegularizer
from models.utils.registry import MODELS, build_module

@MODELS.register_module()
class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True, loss_type: str = "sum",):
        super().__init__()
        self.sample = sample
        self.loss_type = loss_type

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z, self.loss_type)

        log["encoder_latent"] = z
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        if self.loss_type =='mean':
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log
