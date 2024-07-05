# encoding:utf8

import cv2
import numpy as np
import time
from functools import wraps
import os
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from .util import calculate_slices, calculate_latent_slices
import torchvision.transforms as transforms


def read_image(image_path):
    image = cv2.imread(image_path)
    image = imnormalize((image / 255.0).astype(np.float32),
                        np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]), True)

    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    b, c, h, w = image.shape
    # if h % 64 != 0 or w % 64 != 0:
    #     width, height = map(lambda x: x - x % 64, (w, h))
    return image


def get_transform(process_type, short_size, hight, width):
    if process_type == 'resize':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif process_type == 'centercrop':
        transform = transforms.Compose([
            transforms.Resize(short_size, antialias=True),
            transforms.CenterCrop((hight, width)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif process_type == 'centercropwide':
        transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif process_type == 'ori':
        transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize(short_size, antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return transform


def get_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    return vlen, fps


def read_video(video_path, sample_frame, sample_fps, transform, start):
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if sample_fps == -1:
        v_fps = fps
    else:
        v_fps = sample_fps
    if sample_frame == -1:
        acc_samples = vlen
    else:
        acc_samples = min(sample_frame, vlen)
    interval = int(fps / v_fps)
    needed_frames = (acc_samples - 1) * interval + 1
    print('start', start)
    frame_idxs = np.linspace(
        start=start, stop=min(vlen - 1, start + needed_frames), num=acc_samples
    ).astype(int)

    frames = []
    success_idxs = []
    print('frame num', len(frame_idxs))
    print('frame_idxs', frame_idxs)
    frame_id = 0
    max_frame_id = max(frame_idxs)
    while True:
        ret, frame = cap.read()
        if ret and frame_id in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)

            frames.append(frame)
            success_idxs.append(frame_id)
        else:
            pass
        frame_id += 1
        if frame_id > max_frame_id:
            break
    print('frames', len(frames))
    print('video_path:{}, frames:{}, vlen:{}'.format(video_path, frame.shape, vlen))
    frames = torch.stack(frames).float() / 255

    cap.release()

    frames = transform(frames)

    return frames, frame_id


def imdenormalize(img, mean, std, to_bgr=False):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


def imnormalize(img, mean, std, to_rgb=False):
    """Inplace normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x


def add_text_to_frame(frame, text, font_scale=0.6, font_thickness=1, text_color=(255, 255, 255)):
    # 设置字体和文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 获取文本大小
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # 计算文本的位置
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 10  # 偏移量，可以根据需要进行调整
    # 检查文本是否超出图像宽度，需要进行换行
    if text_size[0] > frame.shape[1]:
        # 按照最大宽度计算文本行数
        max_text_width = frame.shape[1]
        # max_chars_per_line = max_text_width // (text_size[0] / len(text))
        max_chars_per_line = len(text) * max_text_width // text_size[0]
        num_lines = len(text) // max_chars_per_line + 1
        chars_per_line = len(text) // num_lines

        # 计算换行后的文本
        lines = []
        for i in range(num_lines):
            if i == num_lines - 1:
                lines.append(text[i * chars_per_line:])
            else:
                lines.append(text[i * chars_per_line: (i + 1) * chars_per_line])

        # 计算每行文本的位置并绘制
        for i, line in enumerate(lines):
            line_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            line_x = (frame.shape[1] - line_size[0]) // 2
            line_y = text_y - (text_size[1] + 5) * (num_lines - i - 1)
            cv2.putText(frame, line, (line_x, line_y), font, font_scale, text_color, font_thickness)
    else:
        # 绘制单行文本
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    return frame


def sliding_window(x, slice_h, slice_w, overlap_h, overlap_w):
    B, T, C, H, W = x.shape
    # 计算窗口实际大小
    window_h = slice_h + overlap_h * 2
    window_w = slice_w + overlap_w * 2

    latent_padding = (overlap_w, overlap_w, overlap_h, overlap_h, 0, 0)
    x = F.pad(x, latent_padding)
    # x = F.pad(x, latent_padding, mode='replicate')

    # 使用 unfold 提取滑动窗口
    # size, step
    windows = x.unfold(3, window_h, window_h - overlap_h * 2).unfold(4, window_w, window_w - overlap_w * 2)
    # 将 windows 重新排列成 (B, T, num_windows_h, num_windows_w, C, slice_h, slice_w)
    # windows torch.Size([1, 9, 3, 1, 2, 576, 480])
    windows = windows.permute(0, 3, 4, 1, 2, 5, 6)  # .contiguous()
    # 调整形状为 (B * num_windows_h * num_windows_w, C, slice_h, slice_w)
    windows = windows.reshape(B, -1, T, C, window_h, window_w)
    return windows


def combine_windows(windows, B, T, C, H, W, slice_h, slice_w, overlap_h, overlap_w):
    num_windows_h = H // slice_h
    num_windows_w = W // slice_w
    combined = torch.zeros((B, T, C, H, W))
    count = torch.zeros((B, T, C, H, W))
    index = 0
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            h_start = i * slice_h
            h_end = h_start + slice_h
            w_start = j * slice_w
            w_end = w_start + slice_w
            combined[:, :, :, h_start: h_end, w_start: w_end] = windows[:, index, :, :, overlap_h:(slice_h + overlap_h),
                                                                overlap_w:(slice_w + overlap_w)]
            # count[:, :, :, (h_start - overlap_h): (h_end + overlap_h), (w_start - overlap_w):(w_end + overlap_w)] += 1
            index += 1
    # combined /= count
    return combined


# video_imgs: t, c, h, w
def sliding_window_sampling(video_imgs, encoder_init_window, encoder_window=17, overlap_frame=0):
    sampled_frames = []
    num_frames = len(video_imgs)
    # 遍历视频帧序列
    if num_frames < encoder_window:
        t, c, h, w = video_imgs.shape
        padding_frame = encoder_window
        padding = torch.zeros((padding_frame, c, h, w))
        sampled_frames.append(torch.cat((video_imgs, padding)))
        return sampled_frames, num_frames
    else:
        sampled_frames.append(video_imgs[:encoder_init_window])
        front_frame = video_imgs[(encoder_init_window - overlap_frame):encoder_init_window]
        print('encoder_init_window, num_frames, encoder_window', encoder_init_window, num_frames, encoder_window)
        if encoder_window > 0:
            for i in range(encoder_init_window, num_frames, encoder_window):
                if overlap_frame > 0:
                    window = video_imgs[i:(i + encoder_window)]
                    sliding_window = torch.concatenate([front_frame, window])
                    front_frame = window[-overlap_frame:]
                    sampled_frames.append(sliding_window)
                else:
                    window = video_imgs[i:(i + encoder_window)]
                    sampled_frames.append(window)
        return sampled_frames, num_frames  # i + read_window_size


def encoder_slice_video_latent(model, silding_window_frame, device, sliding_window_f_id, args):
    # (num_frames, downsample_num, silding_window_frame, encoder_init_window, batch_size, hight, width, slice_h, slice_w, overlap_h, overlap_w, device)
    with torch.no_grad():
        silding_window_num = (args.hight // args.slice_h) * (args.width // args.slice_w)
        T, C, H, W = silding_window_frame.shape
        silding_window_frame = rearrange(silding_window_frame, " (b t) c h w  -> b t c h w", b=args.batch_size)

        silding_window_frame = sliding_window(silding_window_frame, args.slice_h, args.slice_w, args.overlap_h,
                                              args.overlap_w)

        silding_window_frame = rearrange(silding_window_frame, " b s t c h w  -> (b s) t c h w", b=args.batch_size,
                                         s=silding_window_num)

        encode_batch_size = (args.batch_size * silding_window_num) // args.encode_batch_split
        print('encode_batch_size', encode_batch_size)
        batch_z_list = []
        for start_encode_batch_id in range(0, args.batch_size * silding_window_num, encode_batch_size):
            real_encoder_init_window = min(T, args.encoder_init_window)
            batch_silding_window_frame = silding_window_frame[
                                         start_encode_batch_id:(start_encode_batch_id + encode_batch_size)]
            z_list = []
            if sliding_window_f_id == 0:
                is_init_image = True
            else:
                is_init_image = args.encoder_is_init_image

            silding_window_frame_enc = batch_silding_window_frame[:, :real_encoder_init_window]
            silding_window_frame_enc = rearrange(silding_window_frame_enc, "b t c h w  ->  (b t) c h w ",
                                                 b=encode_batch_size)

            init_z, reg_log = model.encode(silding_window_frame_enc,
                                           real_encoder_init_window, is_init_image, return_reg_log=True,
                                           unregularized=args.unregularized)

            init_z = rearrange(init_z, " (b t) c h w -> b t c h w", b=encode_batch_size)

            z_list.append(init_z.detach().cpu())

            torch.cuda.empty_cache()
            del init_z
            overlap_z = (args.encoder_latent_overlap - 1) // args.downsample_num + 1

            v_num_frames = args.encoder_window
            v_num_frames_en_window = args.encoder_window + args.encoder_latent_overlap

            if args.encoder_window > 0 and T > real_encoder_init_window:
                for f_id in range(args.encoder_init_window, v_num_frames, args.encoder_window):
                    slice_f = batch_silding_window_frame[:,
                              (f_id - args.encoder_latent_overlap):(f_id + args.encoder_window)]
                    slice_f = rearrange(slice_f, "b t c h w -> (b t) c h w", b=encode_batch_size)
                    is_init_image = args.encoder_is_init_image
                    slice_z, reg_log = model.encode(slice_f,
                                                    v_num_frames_en_window, is_init_image,
                                                    return_reg_log=True, unregularized=args.unregularized)

                    slice_z = rearrange(slice_z, " (b t) c h w  -> b t c h w",
                                        b=encode_batch_size)
                    slice_z = slice_z[:, overlap_z:]
                    z_list.append(slice_z.detach().cpu())
                    torch.cuda.empty_cache()
                    del slice_z

            z = torch.cat(z_list, dim=1)
            batch_z_list.append(z)

        z = torch.cat(batch_z_list, dim=0)
        return z


def joint_video_slice_latent(model, z, args):
    silding_window_num = (args.hight // args.slice_h) * (args.width // args.slice_w)
    z = rearrange(z, " (b s) t c h w  -> b s t c h w", b=args.batch_size, s=args.batch_size * silding_window_num)
    B, _, latent_T, latent_C, _, _ = z.shape

    latent_H = args.hight // 8
    latent_W = args.width // 8
    latent_slice_h = args.slice_h // 8
    latent_slice_w = args.slice_w // 8
    latent_overlap_h = args.overlap_h // 8
    latent_overlap_w = args.overlap_w // 8

    print(
        f'latent_H: {latent_H}, latent_W: {latent_W}, latent_slice_h: {latent_slice_h}, Slice latent_slice_w: {latent_slice_w}, latent_overlap_h:{latent_overlap_h}, latent_overlap_w:{latent_overlap_w}')
    joint_latent = combine_windows(z, B, latent_T, latent_C, latent_H, latent_W, latent_slice_h, latent_slice_w,
                                   latent_overlap_h,
                                   latent_overlap_w)
    if args.unregularized:
        B, latent_T, latent_C, latent_H, latent_W = joint_latent.shape
        joint_latent = rearrange(joint_latent, "b t c h w -> (b t) c h w", b=B)
        joint_latent, reg_log = model.regularizer(joint_latent)
        joint_latent = rearrange(joint_latent, "(b t) c h w  -> b t c h w ", b=B)
    return joint_latent


def slice_video_latent_decode(model, z, device, sliding_window_f_id, args):
    with torch.no_grad():
        B, latent_T, latent_C, latent_H, latent_W = z.shape

        slice_num_h, slice_num_w, slice_z_h, slice_z_w = calculate_latent_slices(latent_H, latent_W,
                                                                                 max_size=args.max_latent_size,
                                                                                 min_size=args.min_latent_size,
                                                                                 divide=1)
        overlap_z_h = int(0.25 * slice_z_h)
        overlap_z_w = int(0.25 * slice_z_w)
        silding_window_num = slice_num_h * slice_num_w
        print(
            f'slice_num_h: {slice_num_h}, slice_num_w: {slice_num_w}, slice_z_h: {slice_z_h}, slice_z_w: {slice_z_w}, overlap_z_h: {overlap_z_h}, overlap_z_w: {overlap_z_w}')

        decode_batch_split = silding_window_num
        z = sliding_window(z, slice_z_h, slice_z_w, overlap_z_h, overlap_z_w)
        z = rearrange(z, " b s t c h w  -> (b s) t c h w", b=args.batch_size, s=silding_window_num)
        z_t = z.shape[1]

        decode_batch_samples = []

        decode_batch_size = (args.batch_size * silding_window_num) // decode_batch_split
        print('decode_batch_size', decode_batch_size)
        for start_decode_batch_id in range(0, args.batch_size * silding_window_num, decode_batch_size):
            all_samples = []
            batch_z = z[start_decode_batch_id:(start_decode_batch_id + decode_batch_size)]
            init_latent = batch_z[:, :args.decoder_init_window]

            if sliding_window_f_id == 0:
                is_init_image = True
            else:
                is_init_image = args.decoder_is_init_image
            init_latent = rearrange(init_latent, "b t c h w -> (b t) c h w", b=decode_batch_size)

            init_samples = model.decode(init_latent.to(device).to(torch.bfloat16),
                                        decode_batch_size, is_init_image, )

            init_samples = rearrange(init_samples, " (b t) c h w  -> b t c h w", b=decode_batch_size)
            # if is_init_image:
            #     init_samples = init_samples[:, (args.downsample_num - 1 + args.overlap_frame):].detach().cpu()
            # else:
            init_samples = init_samples.detach().cpu()

            print('init_samples', init_samples.shape)
            all_samples.append(init_samples)
            del init_latent
            del init_samples
            torch.cuda.empty_cache()

            if args.decoder_latent_overlap > 0:
                decoder_overlap_f = (args.downsample_num * args.decoder_latent_overlap)
            else:
                decoder_overlap_f = 0

            if args.decoder_window > 0:
                for z_id in range(args.decoder_init_window, z_t, args.decoder_window):
                    slice_z = batch_z[:, (z_id - args.decoder_latent_overlap):(z_id + args.decoder_window)].to(device)
                    slice_z = rearrange(slice_z, "b t c h w -> (b t) c h w", b=decode_batch_size)
                    is_init_image = args.decoder_is_init_image
                    samples = model.decode(slice_z.to(device).to(torch.bfloat16),
                                           decode_batch_size, is_init_image, )
                    samples = rearrange(samples, " (b t) c h w  -> b t c h w", b=decode_batch_size)
                    samples = samples[:, decoder_overlap_f:].detach().cpu()
                    all_samples.append(samples)
                    torch.cuda.empty_cache()
                    del slice_z
                    del samples

            gen_video_frames = torch.cat(all_samples, dim=1)  # b
            print('start_decode_batch_id, gen_video_frames', start_decode_batch_id, gen_video_frames.shape)
            decode_batch_samples.append(gen_video_frames)

        del batch_z
        torch.cuda.empty_cache()

        gen_video_frames = torch.cat(decode_batch_samples, dim=0)
        print('decode_batch_samples gen_video_frames', gen_video_frames.shape)

        gen_video_frames = rearrange(gen_video_frames, " (b s) t c h w  -> b s t c h w", b=args.batch_size,
                                     s=args.batch_size * silding_window_num)
        T = gen_video_frames.shape[2]
        slice_h, slice_w, overlap_h, overlap_w = slice_z_h * args.spatial_downsample_num, slice_z_w * args.spatial_downsample_num, overlap_z_h * args.spatial_downsample_num, overlap_z_w * args.spatial_downsample_num
        gen_video_frames = combine_windows(gen_video_frames, args.batch_size, T, 3, args.hight,
                                           args.width, slice_h, slice_w, overlap_h, overlap_w)

        print('combine_windows gen_video_frames', gen_video_frames.shape)
        return gen_video_frames
