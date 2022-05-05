"""
Approach: "StyleMC: Multi-Channel Based Fast Text-Guided Image Generation and Manipulation"
Reimplemented and modified by Axel Sauer for "StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets"
"""
import os
import re
import click
import legacy
from typing import List, Optional
import PIL.Image
import imageio
from timeit import default_timer as timer

import dnnlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import gen_utils
from metrics.metric_utils import get_feature_detector
from feature_networks import clip

#----------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_image(img, path):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(path)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


#----------------------------------------------------------------------------


def forward_synthesis(G, styles):
    """ pass through synthesis using style space, adjust blocks on the fly if needed """
    G.synthesis.input.affine = nn.Identity()

    x = G.synthesis.input(styles[:, 0][:, :4])

    for idx, name in enumerate(G.synthesis.layer_names):
        block = getattr(G.synthesis, name)
        s = styles[:, idx+1][:, :block.in_channels]

        # adjust block
        block.affine = nn.Identity()
        block.w_dim = block.in_channels

        x = block(x, s)

    if G.synthesis.output_scale != 1:
        x = x * G.synthesis.output_scale

    x = x.to(torch.float32)
    return x


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1:  # Keeps consistent results vs previous method for single objective guidance
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)


def embed_text(model, prompt, device='cuda'):
    return model.encode_text(clip.tokenize(prompt).to(device)).float()


#----------------------------------------------------------------------------


def generate_edit(
    G,
    styles,
    direction,
    edit_strength,
    path,
    save_video=True,
    device='cuda',
):
    time_start = timer()

    if save_video:

        video_out = imageio.get_writer(path+'.mp4', mode='I', fps=60, codec='libx264')
        for grad_change in np.arange(0, 1, 0.005)*edit_strength:

            with torch.no_grad():
                img = forward_synthesis(G, styles + direction*grad_change)

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            video_out.append_data(img[0].to(torch.uint8).cpu().numpy())
        video_out.close()

    else:

        imgs = []
        grad_changes = [x*edit_strength for x in [0, 0.25, 0.5, 0.75, 1]]

        for i, grad_change in enumerate(grad_changes):
            with torch.no_grad():
                img = forward_synthesis(G, styles + direction * grad_change)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            imgs.append(img[0].to(torch.uint8).cpu().numpy())

        PIL.Image.fromarray(np.concatenate(imgs, axis=1), 'RGB').save(path + '.png')

    print(f"Time for generating edits: {timer() - time_start:.2f} s")


def find_direction(
    G,
    styles,
    text_prompt,
    layers,
    seeds,
    class_idx=None,
    batch_size=8,
    device='cuda',
):
    time_start = timer()

    # prepare clip
    detector_url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/feature_networks/clip_vitb_patch32.pkl'
    clip_model = get_feature_detector(url=detector_url).to(device)
    texts = [frase.strip() for frase in text_prompt.split("|") if frase]
    targets = [embed_text(clip_model.model, text, device=device) for text in texts]

    # sample styles
    all_styles = []
    for seed_idx, seed in enumerate(seeds):
        ws = gen_utils.get_w_from_seed(G, 1, device, seed=seed, class_idx=class_idx)
        ws = ws.to(torch.float32).unbind(dim=1)
        all_styles.append(w_s_converter(G, ws, device))
    all_styles = torch.cat(all_styles)

    # stats tracker
    cos_sim_track = AverageMeter('cos_sim', ':.4f')
    norm_track = AverageMeter('norm', ':.4f')
    progress = ProgressMeter(len(seeds), [cos_sim_track, norm_track])

    # initalize styles direction
    direction = torch.zeros(1, G.num_ws, 1024, device=device)
    direction.requires_grad_()
    direction_tracker = torch.zeros_like(direction)

    grads = []
    opt = torch.optim.AdamW([direction], lr=0.05, betas=(0., 0.999), weight_decay=0.25)

    for seed_idx in range(len(seeds)):

        # forward pass through synthesis network with new styles
        styles = all_styles[seed_idx] + direction
        img = forward_synthesis(G, styles)

        # clip loss
        img = (img * 127.5 + 128).clamp(0, 255)
        embeds = clip_model(img)
        cos_sim = prompts_dist_loss(embeds, targets, spherical_dist_loss)
        cos_sim.backward(retain_graph=True)

        # track stats
        cos_sim_track.update(cos_sim.item())
        norm_track.update(torch.norm(direction).item())

        if not (seed_idx % batch_size):

            # zeroing out gradients for non-optimized layers
            layers_zeroed = torch.tensor([x for x in range(G.num_ws) if not x in layers])
            direction.grad[:, layers_zeroed] = 0

            opt.step()
            grads.append(direction.grad.clone())
            direction.grad.data.zero_()

            # keep track of gradients over time
            if seed_idx > 3:
                direction_tracker[grads[-2] * grads[-1] < 0] += 1

            # plot stats
            progress.display(seed_idx)

    # throw out fluctuating channels
    direction = direction.detach()
    direction[direction_tracker > (len(seeds)) / 4] = 0

    print(f"Time for direction search: {timer() - time_start:.2f} s")
    return direction


def w_s_converter(G, ws, device='cuda', unit_test=False):
    styles = torch.zeros(1, G.num_ws, 1024, device=device)

    # input layer
    style = G.synthesis.input.affine(ws[0])
    style_sz = style.shape[1]
    styles[0, :1, :style_sz] = style

    # synthesis layers
    for i, name in enumerate(G.synthesis.layer_names):
        block = getattr(G.synthesis, name)
        w = ws[i+1]
        style = block.affine(w)
        style_sz = style.shape[1]
        styles[0, i+1:i+2, :style_sz] = style

    if unit_test:
        os.makedirs('unit_tests', exist_ok=True)
        with torch.no_grad():
            img_w = G.synthesis(torch.cat(ws)[None])
            img_s = forward_synthesis(G, styles)

        save_image(img_w, 'unit_tests/img_w.png')
        save_image(img_s, 'unit_tests/img_s.png')

    return styles.detach()


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename. Used for computing the direction and optionally producing the final output', required=True)
@click.option('--bigger-network', 'bigger_network_pkl', help='Network pickle filename of bigger network. Used for upsampling.')
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--layers', type=num_range, help='Restrict the style space to a range of layers. We recommend not to optimize the critically sampled layers (last 3).', required=True)
@click.option('--text-prompt', help='Text', type=str, required=True)
@click.option('--edit-strength', help='Strength of edit', type=float, required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True)
@click.option('--class-idx', help='Class label (unconditional if not specified)', type=int)
@click.option('--w-path', help='Path to npz containing style code.', type=str)
@click.option('--init-seed', help='Seed of in inital image', type=int)
@click.option('--save-video', help='Save video of edit', is_flag=True)
def stylemc(
    ctx: click.Context,
    network_pkl: str,
    seeds: List[int],
    layers: List[int],
    text_prompt: str,
    edit_strength: float,
    outdir: str,
    init_seed: Optional[int],
    class_idx: Optional[int],
    w_path: Optional[str],
    save_video: Optional[bool],
    bigger_network_pkl: Optional[str],
):
    assert not((w_path is None) and (init_seed is None)), "Provide either w-path or init-seed"

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # load or sample w
    if w_path:
        w = gen_utils.get_w_from_file(w_path, device)
        w = w.repeat(1, G.num_ws, 1)
    else:
        w = gen_utils.get_w_from_seed(G, batch_sz=1, device=device, truncation_psi=1.0,
                                      seed=init_seed, class_idx=class_idx)

    # find direction
    init_styles = w_s_converter(G, w.unbind(1), device=device)
    direction = find_direction(
        G, init_styles, text_prompt, layers=layers, seeds=seeds, class_idx=class_idx, device=device
    )

    # generate edited images
    if bigger_network_pkl:

        with dnnlib.util.open_url(bigger_network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

        # adjust styles
        w = w[:, 0].repeat(1, G.mapping.num_ws, 1)
        init_styles = w_s_converter(G, w.unbind(1), device=device)
        padding = (0, 0, 0, init_styles.shape[1] - direction.shape[1])
        direction = F.pad(direction, padding, "constant", 0)


    text_prompt = text_prompt.replace(" ", "_")
    path = f'{outdir}/{text_prompt}_{edit_strength}'
    generate_edit(G, init_styles, direction, edit_strength=edit_strength, path=path,
                  save_video=save_video, device=device)

if __name__ == "__main__":
    stylemc()
