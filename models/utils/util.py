
import importlib
from inspect import isfunction
import logging

import mmcv
import torch
import os
import codecs
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_video(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in image_extensions:
        return False
    elif ext in video_extensions:
        return True
    else:
        raise ValueError("The input is not a video or image.")

def calculate_slices(H, W, max_size=512, min_size=256, divide=32):
    def calculate_slices(edge):
        if edge <= 0:
            raise ValueError("The input integer must be positive.")

        closest_diff = float('inf')
        best_slice = 0
        best_slice_num = 0

        for slice_size in range(max_size, min_size - 1, -divide):
            print('edge', edge)
            print('slice_size', slice_size)
            slice_num = edge // slice_size
            total_size = slice_size * slice_num
            diff = abs(edge - total_size)

            if diff < closest_diff:
                closest_diff = diff
                best_slice = slice_size
                best_slice_num = slice_num

        return best_slice, best_slice_num

    slice_h, slice_num_h = calculate_slices(H)
    slice_w, slice_num_w = calculate_slices(W)
    return slice_num_h * slice_h, slice_num_w * slice_w, slice_h, slice_w

def get_logger():
    return logging.getLogger(__name__)

def exists(x):
    return x is not None

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_odd(n):
    return not divisible_by(n, 2)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def is_null_or_empty(x):
    if isinstance(x, list):
        x = x[0]
    if x == u" " or x == u"" or x == b'' or x == b' ' or x == "" or x == " " or x == None or x == "None" or x == "none":
        return True

def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError


def find_all_files(src, abs_path=True, suffixs=None):
    all_files = []
    if suffixs is not None and not isinstance(suffixs, list):
        suffixs = list(suffixs)

    for root, dirs, files in os.walk(src):
        for file in files:
            if suffixs is not None:
                for suffix_i in suffixs:
                    if file.endswith(suffix_i):
                        all_files.append(os.path.join(root, file))
            else:
                all_files.append(os.path.join(root, file))

    if not abs_path:
        all_files = [os.path.relpath(item, src) for item in all_files]
    else:
        all_files = [os.path.abspath(item) for item in all_files]

    return all_files

def get_vids(file_path):
    vid_set = set()
    with codecs.open(file_path, 'r', 'utf-8') as f:
        line = f.readline()
        while not is_null_or_empty(line):
            vid = line.split('.mp4')[0] + '.mp4'
            vid_set.add(vid)
            line = f.readline()
    return list(vid_set)

def calculate_latent_slices(H, W, max_size=40, min_size=24, divide=1):
    def calculate_slices(edge):
        if edge <= 0:
            raise ValueError("The input integer must be a positive integer")

        for slice_size in range(min_size, max_size + 1, divide):
            if edge % slice_size == 0:
                slice_num = edge // slice_size
                return slice_size, slice_num

        raise ValueError(f"No valid slice found for edge={edge} with given constraints")

    slice_h, slice_num_h = calculate_slices(H)
    slice_w, slice_num_w = calculate_slices(W)
    return slice_num_h, slice_num_w, slice_h, slice_w


if __name__=='__main__':
    H, W = 270, 480
    print("H, W", H, W)
    new_H, new_W, slice_h, slice_w = calculate_latent_slices(H, W, max_size=40, min_size=24, divide=1)
    print(f'New Height: {new_H}, New Width: {new_W}, Slice Height: {slice_h}, Slice Width: {slice_w}')

    vid = mmcv.VideoReader('/Users/bornfly/Desktop/hr_video_dataset/BlackMythWukong/gamesci_2024_PV07_EN.mp4')
    print(vid.fps)
