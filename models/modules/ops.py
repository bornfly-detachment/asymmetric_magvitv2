import torch
from einops import rearrange

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )

def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = func(self, x, *args, **kwargs)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x
    return wrapper

def nonlinearity(x):
    return x * torch.sigmoid(x)

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)
