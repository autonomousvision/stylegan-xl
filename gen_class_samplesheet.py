import os
from pathlib import Path
import PIL.Image
from typing import List
import click
import numpy as np
import torch
from tqdm import tqdm

import legacy
import dnnlib
from training.training_loop import save_image_grid
from torch_utils import gen_utils
from gen_images import parse_range

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=42)
@click.option('--centroids-path', type=str, help='Pass path to precomputed centroids to enable multimodal truncation')
@click.option('--classes', type=parse_range, help='List of classes (e.g., \'0,1,4-6\')', required=True)
@click.option('--samples-per-class', help='Samples per class.', type=int, default=4)
@click.option('--grid-width', help='Total width of image grid', type=int, default=32)
@click.option('--batch-gpu', help='Samples per pass, adapt to fit on GPU', type=int, default=32)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
def generate_samplesheet(
    network_pkl: str,
    truncation_psi: float,
    seed: int,
    centroids_path: str,
    classes: List[int],
    samples_per_class: int,
    batch_gpu: int,
    grid_width: int,
    outdir: str,
    desc: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).requires_grad_(False)

    # setup
    os.makedirs(outdir, exist_ok=True)
    desc_full = f'{Path(network_pkl).stem}_trunc_{truncation_psi}'
    if desc is not None: desc_full += f'-{desc}'
    run_dir = Path(gen_utils.make_run_dir(outdir, desc_full))

    print('Generating latents.')
    ws = []
    for class_idx in tqdm(classes):
        w = gen_utils.get_w_from_seed(G, samples_per_class, device, truncation_psi, seed=seed,
                                      centroids_path=centroids_path, class_idx=class_idx)
        ws.append(w)
    ws = torch.cat(ws)

    print('Generating samples.')
    images = []
    for w in tqdm(ws.split(batch_gpu)):
        img = gen_utils.w_to_img(G, w, to_np=True)
        images.append(img)

    # adjust grid widht to prohibit folding between same class then save to disk
    grid_width = grid_width - grid_width % samples_per_class
    images = gen_utils.create_image_grid(np.concatenate(images), grid_size=(grid_width, None))
    PIL.Image.fromarray(images, 'RGB').save(run_dir / 'sheet.png')

if __name__ == "__main__":
    generate_samplesheet()
