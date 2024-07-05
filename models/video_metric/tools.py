
import numpy as np
from ..utils.image_op import trans
from scipy import linalg
import torch
from .fid import load_inception_v3_pretrained, get_fid_score


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert (
            mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
            sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
                  "fid calculation produces singular product; "
                  "adding %s to diagonal of cov estimates"
              ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean



def calculate_fid(features1, features2):
    # 计算第一个数据集的特征向量均值和协方差
    m1 = np.mean(features1, axis=0)
    s1 = np.cov(features1, rowvar=False)
    # 计算第二个数据集的特征向量均值和协方差
    m2 = np.mean(features2, axis=0)
    s2 = np.cov(features2, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def calculate_fvd(real_videos, fake_videos, device, method='styleganv'):
    if method == 'styleganv':
        from scripts.video_metric.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from scripts.video_metric.fvd.videogpt.fvd import load_i3d_pretrained
        from scripts.video_metric.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from scripts.video_metric.fvd.videogpt.fvd import frechet_distance

    print("calculate_fvd...")
    print('assert real_videos.shape == fake_videos.shape', real_videos.shape, fake_videos.shape)

    assert real_videos.shape == fake_videos.shape

    i3d = load_i3d_pretrained(device=device)
    inception_v3 = load_inception_v3_pretrained(device=device)
    real_videos_fvd = trans(real_videos)  # BTCHW -> BCTHW
    fake_videos_fvd = trans(fake_videos)
    fake_embeddings = []
    real_recon_embeddings = []
    fid_list = []
    real_feat_list = []
    fake_feat_list = []
    # print('real_videos_fvd[0]', real_videos_fvd[0].shape) # real_videos_fvd[0] torch.Size([3, 32, 224, 224])
    real_videos_fid = real_videos_fvd[0].permute(1, 0, 2, 3)
    fake_videos_fid = fake_videos_fvd[0].permute(1, 0, 2, 3)

    for v_id, real_videos_feat in enumerate(real_videos_fid):
        # print('real_videos_feat', real_videos_feat.shape)
        real_feat = inception_v3(real_videos_feat.unsqueeze(0).to(device))  # BCTHW
        fake_feat = inception_v3(fake_videos_fid[v_id].unsqueeze(0).to(device))
        real_feat_list.append(real_feat.detach().cpu())
        fake_feat_list.append(fake_feat.detach().cpu())
        torch.cuda.empty_cache()
        del real_feat
        del fake_feat
    real_feat = torch.cat(real_feat_list)
    fake_feat = torch.cat(fake_feat_list)
    fid = calculate_fid(real_feat.numpy(), fake_feat.numpy())

    for clip_timestamp in range(10, real_videos_fvd.shape[-3] + 1):
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        print('clip_timestamp', clip_timestamp)
        videos_clip_real = real_videos_fvd[:, :, : clip_timestamp]
        videos_clip_fake = fake_videos_fvd[:, :, : clip_timestamp]
        real = get_fvd_feats(videos_clip_real, i3d=i3d, device=device)
        fake = get_fvd_feats(videos_clip_fake, i3d=i3d, device=device)
        fake_embeddings.append(real.detach().cpu())
        real_recon_embeddings.append(fake.detach().cpu())

        torch.cuda.empty_cache()
        del real
        del fake

    real_recon_embeddings = torch.cat(real_recon_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)
    fvd = frechet_distance(real_recon_embeddings.clone(), fake_embeddings.clone())
    fvd_star = frechet_distance(real_recon_embeddings.clone(), fake_embeddings.clone())
    return fvd, fvd_star, fid
