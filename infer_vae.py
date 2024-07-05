from models.vae import AsymmetricMagVitV2Pipline
import torch
from models.utils.image_op import imdenormalize, imnormalize, read_video, read_image, sliding_window_sampling, \
    sliding_window, combine_windows, encoder_slice_video_latent, joint_video_slice_latent, slice_video_latent_decode, \
    get_transform, get_video_frame
from models.utils.util import to_torch_dtype, is_video, calculate_slices, str2bool, find_all_files
import numpy as np
import argparse
from einops import rearrange
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

print('torch.__version__', torch.__version__)


def sample(input, v_index, gth_index, gth_path, rec_path, args):
    T, C, H, W = input.shape
    read_h, read_w, slice_h, slice_w = calculate_slices(H, W, max_size=args.max_size, min_size=args.min_size,
                                                        divide=args.divide)

    args.encode_batch_split = (read_h // slice_h) * (read_w // slice_w)

    print(
        f'New Height: {read_h}, New Width: {read_w}, Slice Height: {slice_h}, Slice Width: {slice_w}, encode_batch_split:{args.encode_batch_split}')

    centerCrop = transforms.CenterCrop((read_h, read_w))
    print('input', input.shape)
    input = centerCrop(input)
    print('CenterCrop input', input.shape)
    vis_input = rearrange(input, "t c h w -> t h w c")
    print('vis_input', vis_input.shape)

    for idx, frame in enumerate(vis_input):
        frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
        frame = cv2.cvtColor((frame * 255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(gth_path, "{}.png".format(gth_index)), frame)
        gth_index += 1

    silding_window_frames, last_id = sliding_window_sampling(input, args.encoder_init_window,
                                                             args.encoder_window, args.overlap_frame)
    print('silding_window_frames', len(silding_window_frames))

    args.hight = read_h
    args.width = read_w
    args.slice_h = slice_h
    args.slice_w = slice_w
    args.overlap_h = slice_h // 4
    args.overlap_w = slice_w // 4
    print(f'overlap_h: {args.overlap_h}, overlap_w: {args.overlap_w}')
    print(f'slice_h: {args.slice_h}, slice_w: {args.slice_w}')

    with torch.no_grad():

        for sliding_window_f_id, silding_window_frame in enumerate(silding_window_frames):
            print('sliding_window_f_id, silding_window_frame', sliding_window_f_id, silding_window_frame.shape)
            z = encoder_slice_video_latent(model, silding_window_frame.to(device, dtype), device,
                                           sliding_window_f_id,
                                           args)

            z = joint_video_slice_latent(model, z, args)
            print('encoder_slice_video_latent', z.shape)
            gen_video_frames = slice_video_latent_decode(model, z, device, sliding_window_f_id, args)

            if args.is_vis:
                vis_gen_video_frames = rearrange(gen_video_frames, "b t c h w -> b t h w c").to(torch.float32)
                for idx, frame in enumerate(vis_gen_video_frames[0]):
                    # 1 de normalize, and clip to (0, 1)
                    frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
                    frame = cv2.cvtColor((frame * 255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(rec_path, "{}.png".format(v_index)), frame)
                    v_index += 1
    return v_index, gth_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default='data/gamesci_2024_PV07_EN.mp4')
    parser.add_argument("--output_folder", type=str, default='vae_eval_out/vae_4z_bf16_hf_video_demo2')
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/model_zoo/vae_4z_bf16_hf")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--sample_frame", type=int, default=-1)
    parser.add_argument("--sample_fps", type=int, default=8)
    parser.add_argument("--overlap_frame", type=int, default=0)
    parser.add_argument("--is_vis", type=str2bool, default=True)

    parser.add_argument("--downsample_num", type=int, default=4)
    parser.add_argument("--spatial_downsample_num", type=int, default=8)
    parser.add_argument("--decoder_init_window", type=int, default=5)
    parser.add_argument("--decoder_window", type=int, default=0)
    parser.add_argument("--decoder_latent_overlap", type=int, default=0)

    parser.add_argument("--encoder_init_window", type=int, default=17)
    parser.add_argument("--encoder_window", type=int, default=17)
    parser.add_argument("--encoder_latent_overlap", type=int, default=0)

    parser.add_argument("--encoder_is_init_image", type=str2bool, default=True)
    parser.add_argument("--decoder_is_init_image", type=str2bool, default=True)

    parser.add_argument("--slice_h", type=int, default=8)
    parser.add_argument("--slice_w", type=int, default=8)
    parser.add_argument("--overlap_h", type=int, default=32)
    parser.add_argument("--overlap_w", type=int, default=32)
    parser.add_argument("--slice_z_h", type=int, default=32)
    parser.add_argument("--slice_z_w", type=int, default=32)
    parser.add_argument("--overlap_z_h", type=int, default=2)
    parser.add_argument("--overlap_z_w", type=int, default=2)
    parser.add_argument("--unregularized", type=str2bool, default=True)
    parser.add_argument("--encode_batch_split", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--max_size", type=int, default=512)
    parser.add_argument("--min_size", type=int, default=320)
    parser.add_argument("--divide", type=int, default=32)

    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--min_latent_size", type=int, default=32)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--process_type", type=str, default='ori')
    parser.add_argument("--short_size", type=int, default=256)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--hight", type=int, default=4)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(args.dtype)
    os.makedirs(args.output_folder, exist_ok=True)

    print('args.model_path', args.model_path)
    model = AsymmetricMagVitV2Pipline.from_pretrained(args.model_path).to(device, dtype).eval()
    model.encoder.to(device, dtype).eval()
    model.decoder.to(device, dtype).eval()

    img_transform = get_transform(args.process_type, args.short_size, args.hight, args.width)

    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.flv',
                  '.wmv', '.webm', '.mpeg', '.mpg']
    if os.path.isfile(args.input_path):
        input_paths = [args.input_path]
    else:
        input_paths = find_all_files(args.input_path, suffixs=extensions)

    print('input_paths', input_paths)
    for input_path in tqdm(input_paths):

        v_name = input_path.rsplit('/', 1)[-1]

        video_path = os.path.join(args.output_folder, "{}".format(v_name))
        gth_path = os.path.join(video_path, 'gth')
        rec_path = os.path.join(video_path, 'rec')

        os.makedirs(gth_path, exist_ok=True)
        os.makedirs(rec_path, exist_ok=True)

        is_input_video = is_video(input_path)
        if is_input_video:
            input, last_frame_id = read_video(input_path, args.encoder_init_window, args.sample_fps, img_transform,
                                              start=args.start)
        else:
            input = read_image(input_path)
            args.encoder_init_window = 1
            args.decoder_init_window = 1

        v_index = 0
        gth_index = 0
        v_index, gth_index = sample(input, v_index, gth_index, gth_path, rec_path, args)

        if is_input_video and args.sample_frame == -1:
            vlen, fps = get_video_frame(input_path)

            while last_frame_id < vlen - 1:
                input, last_frame_id = read_video(input_path, args.encoder_init_window, args.sample_fps, img_transform,
                                                  start=last_frame_id)
                if len(input) < args.encoder_init_window or v_index >= 560:
                    break
                v_index, gth_index = sample(input, v_index, gth_index, gth_path, rec_path, args)


