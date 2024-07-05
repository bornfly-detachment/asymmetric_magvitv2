# AsymmetricMagVitV2
Lightweight open-source reproduction of MagVitV2, fully aligned with the paper’s functionality. Supports image and video joint encoding and decoding, as well as videos of arbitrary length and resolution.

* All spatio-temporal operators are implemented using causal 3D to avoid video instability caused by 2D+1D, ensures that the FVD does not sudden increases.
* The Encoder and Decoder support arbitrary resolutions, support auto-regressive inference for arbitrary durations.
* Training employs multi-resolution and dynamic-duration mixed training, allowing decoding of videos with arbitrary odd frames as long as GPU memory permits, demonstrating temporal extrapolation capability.
* The model is closely aligned with MagVitV2 but with reduced parameter, particularly in the lightweight Encoder, reducing the burden of caching VAE features.



## Demo

### 4 channel VAE video reconstruction

#####  video reconstruction

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; padding-right: 5px;">
    <a href="https://upos-sz-mirrorcos.bilivideo.com/upgcxcode/81/23/500001606242381/500001606242381-1-192.mp4?e=ig8euxZM2rNcNbRVhwdVhwdlhWdVhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1720215217&gen=playurlv2&os=cosbv&oi=1385955528&trid=dfdcecbd67244616ba5d87d087fd174bO&mid=0&platform=html5&og=cos&upsig=22ecc0103f02d99f3b71d63876f90e0c&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform,og&bvc=vod&nettype=1&orderid=0,3&buvid=&build=7330300&f=O_0_0&bw=70038&logo=80000000">
      <img src="data/show/gif/vae_4z_bf16_sw_17_wukong.gif" alt="60s 3840x2160" style="width: 100%; height: auto;">
    </a>
  </div>
  <div style="flex: 1; padding-left: 5px;">
    <a href="https://upos-sz-mirrorcos.bilivideo.com/upgcxcode/34/86/500001606228634/500001606228634-1-192.mp4?e=ig8euxZM2rNcNbRghWdVhwdlhWN1hwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1720215268&gen=playurlv2&os=cosbv&oi=2584261250&trid=b9c2bd6cb8274f6bb7379d657de5d9e8O&mid=0&platform=html5&og=cos&upsig=c3b7842eea1adc4773ab675f2fcd5d21&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform,og&bvc=vod&nettype=1&orderid=0,3&buvid=&build=7330300&f=O_0_0&bw=121081&logo=80000000">
      <img src="data/show/gif/vae_4z_bf16_sw_17_tokyo_walk_h264_16s.gif" alt="60s 1920x1080" style="width: 100%; height: auto;">
    </a>
  </div>
</div>


* Converting MP4 to GIF may result in detail loss, pixelation, and incomplete duration. It is recommended to watch the original video for the best experience.

######  60s 3840x2160

[bilibili_Black Myth：Wu KongULR 4zVAE](https://www.bilibili.com/video/BV1mjaPe8EWn/?spm_id_from=333.999.0.0&vd_source=681432e843390b0f7192d64fa4ed9613)



######  60s 1920x1080

[bilibili_tokyo_walk ULR 4zVAE](https://www.bilibili.com/video/BV1cCaceLEiq/?t=8.7&vd_source=681432e843390b0f7192d64fa4ed9613)



##### image reconstruction

<table>
  <tr>
    <td><img src="data/show/images/4z/mj_1.png" alt="1" style="width:100%;"></td>
    <td><img src="data/show/images/4z/mj_2.png" alt="2" style="width:100%;"></td>
    <td><img src="data/show/images/4z/mj_3.png" alt="3" style="width:100%;"></td>
  </tr>
  <tr>
    <td><img src="data/show/images/4z/mj_4.png" alt="4" style="width:100%;"></td>
    <td><img src="data/show/images/4z/mj_5.png" alt="5" style="width:100%;"></td>
    <td><img src="data/show/images/4z/mj_6.png" alt="6" style="width:100%;"></td>
  </tr>
  <tr>
    <td><img src="data/show/images/4z/mj_7.png" alt="7" style="width:100%;"></td>
    <td><img src="data/show/images/4z/mj_8.png" alt="8" style="width:100%;"></td>
    <td><img src="data/show/images/4z/mj_9.png" alt="9" style="width:100%;"></td>
  </tr>
</table>

* The original images are located in data/images




### 16 channel VAE image reconstruction


<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; padding-right: 5px;">
    <a href="https://upos-sz-mirrorbd.bilivideo.com/upgcxcode/49/26/500001606242649/500001606242649-1-192.mp4?e=ig8euxZM2rNcNbRVhwdVhwdlhWdVhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1720214786&gen=playurlv2&os=bdbv&oi=2584261250&trid=bbf4c7694b334e96b66f466568155cfbO&mid=0&platform=html5&og=hw&upsig=1fb8442f381fb7d42fce1f234299118e&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform,og&bvc=vod&nettype=1&orderid=0,3&buvid=&build=7330300&f=O_0_0&bw=71316&logo=80000000">
      <img src="data/show/gif/vae_16z_bf16_sw_17_wukong.gif" alt="60s 3840x2160" style="width: 100%; height: auto;">
    </a>
  </div>
  <div style="flex: 1; padding-left: 5px;">
    <a href="https://upos-sz-mirrorcos.bilivideo.com/upgcxcode/20/29/500001606242920/500001606242920-1-192.mp4?e=ig8euxZM2rNcNbRghwdVhwdlhWNVhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1720214870&gen=playurlv2&os=cosbv&oi=2584261250&trid=19d089cf9c3c402c83d4608511c50f60O&mid=0&platform=html5&og=cos&upsig=faeda47e09a67f389cf8ec67c7e0c17c&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform,og&bvc=vod&nettype=1&orderid=0,3&buvid=&build=7330300&f=O_0_0&bw=120172&logo=80000000">
      <img src="data/show/gif/vae_16z_bf16_sw_17_tokyo_walk_h264_16s.gif" alt="60s 1920x1080" style="width: 100%; height: auto;">
    </a>
  </div>
</div>

* Converting MP4 to GIF may result in detail loss, pixelation, and incomplete duration. It is recommended to watch the original video for the best experience.

######  60s 3840x2160

[bilibili_Black Myth：Wu KongULR 16zVAE](https://www.bilibili.com/video/BV1ULaPecEga/?spm_id_from=333.999.0.0&vd_source=681432e843390b0f7192d64fa4ed9613)

######  60s 1920x1080

[bilibili_tokyo_walk ULR 16zVAE](https://www.bilibili.com/video/BV1mLaPecEXP/?spm_id_from=333.999.0.0&vd_source=681432e843390b0f7192d64fa4ed9613)


##### image reconstruction

<table>
  <tr>
    <td><img src="data/show/images/16z/mj_16z_1.png" alt="1" style="width:100%;"></td>
    <td><img src="data/show/images/16z/mj_16z_2.png" alt="2" style="width:100%;"></td>
    <td><img src="data/show/images/16z/mj_16z_3.png" alt="3" style="width:100%;"></td>
  </tr>
  <tr>
    <td><img src="data/show/images/16z/mj_16z_4.png" alt="4" style="width:100%;"></td>
    <td><img src="data/show/images/16z/mj_16z_5.png" alt="5" style="width:100%;"></td>
    <td><img src="data/show/images/16z/mj_16z_6.png" alt="6" style="width:100%;"></td>
  </tr>
  <tr>
    <td><img src="data/show/images/16z/mj_16z_7.png" alt="7" style="width:100%;"></td>
    <td><img src="data/show/images/16z/mj_16z_8.png" alt="8" style="width:100%;"></td>
    <td><img src="data/show/images/16z/mj_16z_9.png" alt="9" style="width:100%;"></td>
  </tr>
</table>


## Contents

- [Installation](#installation)
- [Model Weights](#model-weights)
- [Metric](#metric)
- [Inference](#inference)
- [TODO List](#1)
- [Contact Us](#2)
- [Reference](#3)

### Installation

<a name="installation"></a>

#### 1. Clone the repo

```shell
git clonehttps://github.com/bornfly-detachment/AsymmetricMagVitV2.git
cd AsymmetricMagVitV2
```

#### 2. Setting up the virtualenv

This is assuming you have navigated to the `AsymmetricMagVitV2` root after cloning it.


```shell
# install required packages from pypi
python3 -m venv .pt2
source .pt2/bin/activate
pip3 install -r requirements/pt2.txt
```


### Model Weights

<details>
<summary>View more</summary>

| model | downsample (THW)         | Encoder Size | Decoder Size|
|--------|--------|------|------|
|svd 2Dvae|1x8x8|34M|64M|
|AsymmetricMagVitV2|4x8x8|100M|159M|


| model                  | Data         | #iterations | URL                                                                   |
|------------------------|--------------|-------------|-----------------------------------------------------------------------|
| AsymmetricMagVitV2_4z  |20M Intervid  | 2node 1200k | [AsymmetricMagVitV2_4z](https://huggingface.co/BornFlyReborn/AsymmetricMagVitV2_4z)  |
| AsymmetricMagVitV2_16z |20M Intervid  | 4node 860k  | [AsymmetricMagVitV2_16z](https://huggingface.co/BornFlyReborn/AsymmetricMagVitV2_16z) |


</details>

### Metric

<a name="Metric"></a>

|model|temporal-frame| fvd       |fid|psnr|ssim|
|-----|----|-----------|--|----|----|
|SVD VAE|1 | 190.6  |1.8|28.2|1.0|
|openSoraPlan|1 | 249.8     |1.04|29.6|0.99|
|openSoraPlan|17 | 725.4     |3.17|23.4|0.89|
|openSoraPlan|33 | 756.8     |3.5|23|0.89|
|AsymmetricMagVitV2_4z|1 | 113.5     |1.4|29.8|1.0|
|AsymmetricMagVitV2_4z|17 | 278.5     |2.3|26.4|1.0|
|AsymmetricMagVitV2_4z|33 | 293.3     |2.5|26.3|1.0|
|AsymmetricMagVitV2_16z|1 | 106.7     |0.2|36.3|1.0|
|AsymmetricMagVitV2_16z|17 | 131.4     |0.8|30.7|1.0|
|AsymmetricMagVitV2_16z|33 | 208.2     |1.4|28.9|1.0|

Note: 
1. The test video is the original scale of data/videos/tokyo_walk.mp4. Previously, preprocessing with resize+CenterCrop256 
resolution was also tested on a larger test set, and the results showed consistent trends. Now, it has been found 
that high-resolution and original-sized videos pose the most challenging task for 3DVAE. Therefore, only this one video was tested,
configured at 8fps, and evaluated for the first 10 seconds.
2. The evaluation code can be referenced in models/evaluation.py. However, it has been a while since I last ran it, 
and there have been modifications to the inference code. Calculating FID and FVD scores depends on the model, 
original image preprocessing, inference hyperparameters, and the randomness introduced by sampling encoder KL. 
As a result, scores cannot be accurately reproduced. Nonetheless, this can serve as a reference for designing 
one’s own benchmark.



### Inference

#### About Encoder hyperparameter configuration
* slice frame spatial using： --max_siz --min_size
* slice video temporal using： --encoder_init_window --encoder_window

If the GPU VRAM is not sufficient, metrics for evaluation can be adjusted to be between 256 and 512 at maximum.

#### About Decoder hyperparameter configuration

 * slice latent spatial using： --min_latent_size --max_latent_size
 
(default GPU VRAM needs to exceed 28GB. If the GPU VRAM is not sufficient, metrics for evaluation can be adjusted to be between 32=256p/8 and 64=512p/8 at maximum.)

 * slice latent temporal using： --decoder_init_window,
 
5 frames of latent space corresponds to 17 frames of the original video.
The calculation formula is as follows: latent_T_dim = (frame_T_dim - 1) / temporal_downsample_num;  in this model, temporal_downsample_num=4


#### 1. encode & decode video

```shell
python infer_vae.py --input_path data/videos/tokyo-walk.mp4 --model_path vae_16z_bf16_hf  --output_folder vae_eval_out/vae_4z_bf16_hf_videos > infer_vae_video.log 2>&1  
```

#### 2. encode & decode image

```shell
python infer_vae.py --input_path data/images --model_path vae_16z_bf16_hf  --output_folder vae_eval_out/vae_4z_bf16_hf_images > infer_vae_image.log 2>&1  
```


### TODO List

<p id="1"></p> 

* Reproducing Sora, a 16-channel VAE integrated with SD3. Due to limited computational resources, the focus is on generating 1K high-definition dynamic wallpapers.

* Reproducing VideoPoet, supporting multimodal joint representation. Due to limited computational resources, the focus is on generating music videos.


### Contact Us

<p id="2"></p> 

1. If there are any code-related questions, feel free to contact me via email——bornflyborntochange@outlook.com.
2. You need to scan the image to join the WeChat group or if it is expired, add this student as a friend first to invite you.
<img src="data/assets/mmqrcode1720196270375.png" alt="ding group" width="30%"/>



### Reference

<p id="3"></p> 

- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- SVD: https://github.com/Stability-AI/generative-models
