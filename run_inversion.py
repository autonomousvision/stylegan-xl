"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import dill
import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from tqdm import trange
import dnnlib
import legacy
from metrics import metric_utils
import timm

from training.diffaug import DiffAugment
from pg_modules.blocks import Interpolate


def get_morphed_w_code(new_w_code, fixed_w, regularizer_alpha=30):
    interpolation_direction = new_w_code - fixed_w
    interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
    direction_to_move = regularizer_alpha * interpolation_direction / interpolation_direction_norm
    result_w = fixed_w + direction_to_move
    return result_w


def space_regularizer_loss(
    G_pti,
    G_original,
    w_batch,
    vgg16,
    num_of_sampled_latents=1,
    lpips_lambda=10,
):

    z_samples = np.random.randn(num_of_sampled_latents, G_original.z_dim)
    z_samples = torch.from_numpy(z_samples).to(w_batch.device)

    if not G_original.c_dim:
        c_samples = None
    else:
        c_samples = F.one_hot(torch.randint(G_original.c_dim, (num_of_sampled_latents,)), G_original.c_dim)
        c_samples = c_samples.to(w_batch.device)

    w_samples = G_original.mapping(z_samples, c_samples, truncation_psi=0.5)
    territory_indicator_ws = [get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

    for w_code in territory_indicator_ws:
        new_img = G_pti.synthesis(w_code, noise_mode='none', force_fp32=True)
        with torch.no_grad():
            old_img = G_original.synthesis(w_code, noise_mode='none', force_fp32=True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if new_img.shape[-1] > 256:
            new_img = F.interpolate(new_img, size=(256, 256), mode='area')
            old_img = F.interpolate(old_img, size=(256, 256), mode='area')

        new_feat = vgg16(new_img, resize_images=False, return_lpips=True)
        old_feat = vgg16(old_img, resize_images=False, return_lpips=True)
        lpips_loss = lpips_lambda * (old_feat - new_feat).square().sum()

    return lpips_loss / len(territory_indicator_ws)


def pivotal_tuning(
    G,
    w_pivot,
    target,
    device: torch.device,
    num_steps=350,
    learning_rate = 3e-4,
    noise_mode="const",
    verbose = False,
):
    G_original = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    G_pti = copy.deepcopy(G).train().requires_grad_(True).to(device)
    w_pivot.requires_grad_(False)

    # Load VGG16 feature detector.
    vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, device=device)

    # l2 criterion
    l2_criterion = torch.nn.MSELoss(reduction='mean')

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # initalize optimizer
    optimizer = torch.optim.Adam(G_pti.parameters(), lr=learning_rate)

    # run optimization loop
    all_images = []
    for step in range(num_steps):
        # Synth images from opt_w.
        synth_images = G_pti.synthesis(w_pivot[0].repeat(1,G.num_ws,1), noise_mode=noise_mode)

        # track images
        synth_images = (synth_images + 1) * (255/2)
        synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        all_images.append(synth_images_np)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # LPIPS loss
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        lpips_loss = (target_features - synth_features).square().sum()

        # MSE loss
        mse_loss = l2_criterion(target_images, synth_images)

        # space regularizer
        reg_loss = space_regularizer_loss(G_pti, G_original, w_pivot, vgg16)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss = mse_loss + lpips_loss + reg_loss
        loss.backward()
        optimizer.step()

        msg  = f'[ step {step+1:>4d}/{num_steps}] '
        msg += f'[ loss: {float(loss):<5.2f}] '
        msg += f'[ lpips: {float(lpips_loss):<5.2f}] '
        msg += f'[ mse: {float(mse_loss):<5.2f}]'
        msg += f'[ reg: {float(reg_loss):<5.2f}]'
        if verbose: print(msg)

    return all_images, G_pti


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps = 1000,
    w_avg_samples = 10000,
    initial_learning_rate = 0.1,
    lr_rampdown_length = 0.25,
    lr_rampup_length = 0.05,
    verbose = False,
    device: torch.device,
    noise_mode="const",
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = torch.from_numpy(np.random.RandomState(123).randn(w_avg_samples, G.z_dim)).to(device)

    # get class probas by classifier
    if not G.c_dim:
        c_samples = None
    else:
        classifier = timm.create_model('deit_base_distilled_patch16_224', pretrained=True).eval().to(device)
        cls_target = F.interpolate((target.to(device).to(torch.float32) / 127.5 - 1)[None], 224)
        logits = classifier(cls_target).softmax(1)
        classes = torch.multinomial(logits, w_avg_samples, replacement=True).squeeze()
        print(f'Main class: {logits.argmax(1).item()}, confidence: {logits.max().item():.4f}')
        c_samples = np.zeros([w_avg_samples, G.c_dim], dtype=np.float32)
        for i, c in enumerate(classes):
            c_samples[i, c] = 1
        c_samples = torch.from_numpy(c_samples).to(device)

    w_samples = G.mapping(z_samples, c_samples)  # [N, L, C]

    # get empirical w_avg
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]

    # Load VGG16 feature detector.
    vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, device=device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # initalize optimizer
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # run optimization loop
    all_images = []
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        synth_images = G.synthesis(w_opt[0].repeat(1,G.num_ws,1), noise_mode=noise_mode)

        # track images
        synth_images = (synth_images + 1) * (255/2)
        synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        all_images.append(synth_images_np)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        lpips_loss = (target_features - synth_features).square().sum()

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss = lpips_loss
        loss.backward()
        optimizer.step()
        msg  = f'[ step {step+1:>4d}/{num_steps}] '
        msg += f'[ loss: {float(loss):<5.2f}] '
        if verbose: print(msg)

    return all_images, w_opt.detach()[0]


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--seed', help='Random seed', type=int, default=42, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--inv-steps', help='Number of inversion steps', type=int, default=1000, show_default=True)
@click.option('--w-init', help='path to inital latent', type=str, default='', show_default=True)
@click.option('--run-pti', help='run pivotal tuning', is_flag=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=350, show_default=True)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    inv_steps: int,
    w_init: str,
    run_pti: bool,
    pti_steps: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Latent optimization
    start_time = perf_counter()
    all_images = []
    if not w_init:
        print('Running Latent Optimization...')
        all_images, projected_w = project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
            num_steps=inv_steps,
            device=device,
            verbose=True,
            noise_mode='const',
        )
        print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')
    else:
        projected_w = torch.from_numpy(np.load(w_init)['w'])[0].to(device)

    start_time = perf_counter()

    # Run PTI
    if run_pti:
        print('Running Pivotal Tuning Inversion...')
        gen_images, G = pivotal_tuning(
            G,
            projected_w,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
            device=device,
            num_steps=pti_steps,
            verbose=True,
        )
        all_images += gen_images
        print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for synth_image in all_images:
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    synth_image = G.synthesis(projected_w.repeat(1, G.num_ws, 1))
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')

    # save latents
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    # Save Generator weights
    snapshot_data = {'G': G, 'G_ema': G}
    with open(f"{outdir}/G.pkl", 'wb') as f:
        dill.dump(snapshot_data, f)

    #----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter
