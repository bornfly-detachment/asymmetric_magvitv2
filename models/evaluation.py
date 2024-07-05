
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from typing import Optional
from safetensors.torch import load_file as load_safetensors
import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from utils.util import get_vids, find_all_files, to_torch_dtype, str2bool, default, calculate_slices
from utils.image_op import sliding_window_sampling, sliding_window, combine_windows, read_video, encoder_slice_video_latent, joint_video_slice_latent, slice_video_latent_decode
import codecs
from piq import psnr, ssim
import collections
from video_metric.tools import calculate_fvd
from ..models.vae import AsymmetricMagVitV2Pipline
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, root_dir, short_size=128, transform=None, seg=10, fps=8, sample_frame=16, device='cuda', dtype='torch.bfloat16'):
        self.root_dir = root_dir
        self.transform = transform
        self.seg = seg
        self.fps = fps
        self.device = device
        self.dtype = dtype
        self.short_size = short_size
        if os.path.isfile(root_dir):
            if root_dir.endswith('.mp4'):
                self.video_files = [root_dir]
            else:
                self.video_files = get_vids(root_dir)

        else:
            self.video_files = find_all_files(root_dir, suffixs=['.mp4'])

        if sample_frame == -1:
            self.sample_frame = float('inf')
        else:
            self.sample_frame = sample_frame
        print('self.video_files', len(self.video_files))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):

        # try:
        video_path = self.video_files[idx]
        v_name = video_path.rsplit('/', 1)[-1]
        video = read_video(video_path, self.sample_frame, self.fps) #self.read_video(video_path)
        video = self.transform(video)
        # except Exception as e:
        #     # 如果发生了其他异常，执行这个代码块
        #     print("发生了一个异常:", e)
        #     video_path = os.path.join(self.root_dir, self.video_files[(idx+1)%len(self.video_files)])
        #     v_name = video_path.rsplit('/', 1)[-1]
        #     video = self.read_video(video_path, self.process_type)
        #     return video, v_name

        return video, v_name



def sample(
        input_path: str = "/mnt/workspace/ai-story/public/datasets/k600/demo_test",
        num_frames: Optional[int] = None,
        save_eval_file='eval_vae.log',
        overlap_frame=1,
        batch_size=1,
        num_steps: Optional[int] = None,
        version: str = "svd",
        sample_frame=16,
        width: int = 512,
        hight: int = 512,
        seed: int = 23,
        is_metric: bool = True,
        device: str = "cuda",
        output_folder: Optional[str] = None,
        ckpt_path='',
        config_file='',
        is_vis=False,
        seg=10,
        fps=8,
        process_type='resize_ori',
        short_size=512,
        args=None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 40)
        num_steps = default(num_steps, 25)
        model_config = config_file
    else:
        raise ValueError(f"Version {version} does not exist.")

    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(args.dtype)
    os.makedirs(args.output_folder, exist_ok=True)

    model = AsymmetricMagVitV2Pipline.from_pretrained(args.model_path).to(device, dtype).eval()
    model.encoder.to(device, dtype).eval()
    model.decoder.to(device, dtype).eval()

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

    dataset = VideoDataset(root_dir=input_path, short_size=short_size, fps=fps, transform=transform, seg=seg,
                           sample_frame=sample_frame, device=device, dtype=dtype)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    fvds = []
    fvds_star = []
    fid_list = []
    psnr_list = []
    ssim_list = []
    latent_mean_list = []
    latent_var_list = []

    # 使用 DataLoader 迭代数据

    for batch_idx, item in tqdm(enumerate(data_loader)):
        with open(save_eval_file, 'a+') as f:
            batch_video_imgs, v_name = item  # b, t, c, h, w
            B, T, C, H, W = batch_video_imgs.shape
            print('video_imgs', batch_video_imgs.shape)

            all_samples = []
            #  b, t, c, h, w ——> b, slice, t, c, h, w

            for batch_id, video_imgs in enumerate(batch_video_imgs):
                v_name = v_name[batch_id]
                video_path = os.path.join(output_folder, "{}".format(v_name))
                gen_path = os.path.join(video_path, 'gen')
                rec_path = os.path.join(video_path, 'rec')
                os.makedirs(video_path, exist_ok=True)
                os.makedirs(gen_path, exist_ok=True)
                os.makedirs(rec_path, exist_ok=True)

                read_h, read_w, slice_h, slice_w = calculate_slices(H, W, max_size=args.max_size,
                                                                    min_size=args.min_size,
                                                                    divide=args.divide)

                print(
                    f'New Height: {read_h}, New Width: {read_w}, Slice Height: {slice_h}, Slice Width: {slice_w}, encode_batch_split:{args.encode_batch_split}')
                input = F.interpolate(input, size=(read_h, read_w), mode='bilinear',
                                      align_corners=False)

                silding_window_frames, last_id = sliding_window_sampling(video_imgs, args.encoder_init_window,
                                                                         args.encoder_window, args.overlap_frame)
                gen_video_frames_sliding_window = []
                v_index = 0
                with torch.no_grad():
                    for sliding_window_f_id, silding_window_frame in enumerate(silding_window_frames):
                        print('sliding_window_f_id, silding_window_frame', sliding_window_f_id, silding_window_frame.shape)

                        z = encoder_slice_video_latent(model, silding_window_frame.to(device, dtype), device, sliding_window_f_id, args)

                        z = joint_video_slice_latent(model, z, args)
                        torch.cuda.empty_cache()
                        v_mean = torch.mean(
                            z.detach().cpu())  # reg_log["mean"].detach().cpu()   # torch.mean(z.detach().cpu())
                        v_var = torch.var(
                            z.detach().cpu())  # reg_log["var_pred"].detach().cpu()  #torch.var(z.detach().cpu())
                        latent_mean_list.append(v_mean)
                        latent_var_list.append(v_var)

                        print('slice_video_latent_decode z', z.shape)
                        gen_video_frames = slice_video_latent_decode(model, z, device, sliding_window_f_id, args)
                        print('slice_video_latent_decode gen_video_frames', gen_video_frames.shape)  # 1, 1, 3, 1024, 1920
                        gen_video_frames_sliding_window.append(gen_video_frames)

                        if is_vis:
                            vis_gen_video_frames = rearrange(gen_video_frames, "b t c h w -> b t h w c").to(torch.float32)
                            for idx, frame in enumerate(vis_gen_video_frames[0]):
                                # 1 de normalize, and clip to (0, 1)
                                frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
                                frame = cv2.cvtColor((frame * 255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(gen_path, "{}.png".format(v_index)), frame)
                                v_index += 1

                        torch.cuda.empty_cache()
                        del gen_video_frames
                        del z

                print('gen_video_frames_sliding_window', len(gen_video_frames_sliding_window))

                fvd_video_imgs = video_imgs
                vis_fvd_video_imgs = rearrange(fvd_video_imgs[0], "t c h w -> t h w c")

                gen_video_frames = torch.cat(gen_video_frames_sliding_window, dim=1)

                if is_vis:
                    for idx, frame in enumerate(vis_fvd_video_imgs):
                        frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
                        frame = cv2.cvtColor((frame * 255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(rec_path, "{}.png".format(idx)), frame)

                try:
                    if is_metric:
                        fvd, fvd_star, fid = calculate_fvd(fvd_video_imgs.to(torch.float32),
                                                           gen_video_frames.to(torch.float32),
                                                           device=device, method='videogpt')
                        fvds.append(fvd)
                        fvds_star.append(fvd_star)
                        fid_list.append(fid)
                        fvd_video_imgs = torch.clamp((fvd_video_imgs + 1.0) / 2.0, min=0.0, max=1.0)
                        gen_video_frames = torch.clamp((gen_video_frames + 1.0) / 2.0, min=0.0, max=1.0)

                        psnr_value = psnr(fvd_video_imgs.squeeze(0).to(torch.float32),
                                          gen_video_frames.squeeze(0).to(torch.float32), data_range=1.0)
                        ssim_value = ssim(fvd_video_imgs.squeeze(0).to(torch.float32),
                                          gen_video_frames.squeeze(0).to(torch.float32), data_range=1.0)
                        psnr_list.append(psnr_value)
                        ssim_list.append(ssim_value)

                        # psnr_value, ssim_value = 0, 0
                        out_line = "vid:{}, fvd:{}, fid:{}, v_mean:{}, v_var:{} psnr:{}, ssim:{}".format(v_name, fvd, fid,
                                                                                                         v_mean, v_var,
                                                                                                         psnr_value,
                                                                                                         ssim_value)
                        print(out_line)
                        f.write(out_line + '\n')
                        f.close()
                        torch.cuda.empty_cache()
                        del fvd_video_imgs
                        del gen_video_frames
                except Exception as e:
                    print(f"metric fvd error: {e}")

    if is_metric:
        fvd_mean = np.mean(fvds)
        fvd_std = np.std(fvds)
        fvd_star_mean = np.mean(fvds_star)
        fvd_star_std = np.std(fvds_star)
        psnr_mean = np.mean(psnr_list)
        psnr_std = np.std(psnr_list)
        ssim_mean = np.mean(ssim_list)
        ssim_std = np.std(ssim_list)
        fid_mean = np.mean(fid_list)
        fid_std = np.std(fid_list)
        latent_mean = np.mean(latent_mean_list)
        latent_var = np.mean(latent_var_list)

        with open(save_eval_file, 'a+') as f:
            # print(f" FID {fid_mean:.2f} +/- {fid_std:.2f} FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/0 {fvd_star_std:.2f}")
            # f.write(f" FID {fid_mean:.2f} +/- {fid_std:.2f} FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/0 {fvd_star_std:.2f}" + '\n')
            f.write(
                f"latent_mean:{latent_mean:.1}, latent_var:{latent_var:.1f},  FID {fid_mean:.1f} ± {fid_std:.1f}, FVD {fvd_mean:.1f} ± {fvd_std:.1f}, PSNR {psnr_mean:.1f} ± {psnr_std:.1f}, SSIM {ssim_mean:.1f} ± {ssim_std:.1f}" + '\n')
            f.close()


def model_load_ckpt(model, path):
    # TODO: how to load ema weights?
    if path.endswith("ckpt") or path.endswith("pt"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
    elif path.endswith("safetensors"):
        sd = load_safetensors(path)
    else:
        raise NotImplementedError(f"Unknown checkpoint format: {path}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
    )
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected Keys: {unexpected}")
    return model




if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='/mnt/data/ai-story/zhuoqun.luo/model_zoo/svd.safetensors')
    parser.add_argument("--output_folder", type=str, default='vae_eval_out/svd_vae_baseline')
    parser.add_argument("--save_eval_file", type=str, default='vae_eval_out/svd_vae_baseline_eval.log')
    parser.add_argument("--config_file", type=str,
                        default='/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/svd_vae_infer.yaml')
    parser.add_argument("--sample_frame", type=int, default=32)

    # parser.add_argument("--ckpt_path", type=str, default='/mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/logs/2024-04-09T20-10-18_video_vae_config-causal_vae_tail_v6_4node/checkpoints/epoch=000002.ckpt')
    # parser.add_argument("--output_folder", type=str, default='vae_eval_out/causal_vae_tail_v6_4node')
    # parser.add_argument("--config_file", type=str, default='/mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/configs/video_vae_config/causal_vae_tail_v6_4node.yaml')
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--seg", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)

    parser.add_argument("--image_repeat_num", type=int, default=8)
    parser.add_argument("--decoder_latent_num", type=int, default=2)
    parser.add_argument("--is_encode_2d", type=str2bool, default=False)

    parser.add_argument("--overlap_frame", type=int, default=0)
    parser.add_argument("--is_vis", type=str2bool, default=False)
    parser.add_argument("--is_causal", type=str2bool, default=False)
    # parser.add_argument("--is_resize", action='store_true', default=False)
    parser.add_argument("--process_type", type=str, default='resize_ori')
    parser.add_argument("--resize_method", type=str, default='box')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_path", type=str,
                        default="/mnt/workspace/ai-story/zhuoqun.luo/dataset/mini_vae_test/demo")
    parser.add_argument("--short_size", type=int, default=256)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--hight", type=int, default=4)

    parser.add_argument("--downsample_num", type=int, default=4)
    parser.add_argument("--spatial_downsample_num", type=int, default=8)

    parser.add_argument("--decoder_latent_start_id", type=int, default=0)
    parser.add_argument("--is_dynamic", type=str2bool, default=False)
    parser.add_argument("--is_metric", type=str2bool, default=False)


    parser.add_argument("--decoder_init_window", type=int, default=4)
    parser.add_argument("--decoder_window", type=int, default=4)
    parser.add_argument("--decoder_latent_overlap", type=int, default=4)

    parser.add_argument("--encoder_init_window", type=int, default=17)
    parser.add_argument("--encoder_window", type=int, default=8)
    parser.add_argument("--encoder_latent_overlap", type=int, default=8)
    parser.add_argument("--encoder_is_init_image", type=str2bool, default=False)
    parser.add_argument("--decoder_is_init_image", type=str2bool, default=False)

    parser.add_argument("--slice_h", type=int, default=8)
    parser.add_argument("--slice_w", type=int, default=8)
    parser.add_argument("--overlap_h", type=int, default=8)
    parser.add_argument("--overlap_w", type=int, default=8)
    parser.add_argument("--unregularized", type=str2bool, default=False)

    parser.add_argument("--max_size", type=int, default=320)
    parser.add_argument("--min_size", type=int, default=256)
    parser.add_argument("--divide", type=int, default=32)

    parser.add_argument("--max_latent_size", type=int, default=48)
    parser.add_argument("--min_latent_size", type=int, default=24)
    parser.add_argument("--max_size", type=int, default=512)
    parser.add_argument("--min_size", type=int, default=320)
    parser.add_argument("--divide", type=int, default=32)



    args = parser.parse_args()

    # '/mnt/workspace/ai-story/zhuoqun.luo/workspace/exp_2dplus1d_v3/ti2v/configs/video_vae_config/v4_8z.yaml'

    sample(input_path=args.input_path,
           num_frames=args.num_frames,
           is_metric=args.is_metric,
           save_eval_file=args.save_eval_file,
           num_steps=None,
           width=args.width,
           hight=args.hight,
           version="svd",
           process_type=args.process_type,
           sample_frame=args.sample_frame,
           fps=args.fps,
           seed=23,
           seg=args.seg,
           device="cuda",
           is_vis=args.is_vis,
           overlap_frame=args.overlap_frame,
           short_size=args.short_size,
           output_folder=args.output_folder,
           ckpt_path=args.ckpt_path,
           config_file=args.config_file,
           args=args)
