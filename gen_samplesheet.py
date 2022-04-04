import os
from pathlib import Path
import click
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import legacy
import dnnlib
from training.training_loop import save_image_grid
from torch_utils import gen_utils

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=42)
@click.option('--max-classes', help='Limit max number of classes', type=int, default=1000)
@click.option('--samples-per-class', help='Samples per class (for unimodal dataset cls size is 1).', type=int, default=4)
@click.option('--classes-per-row', help='Classes per row.', type=int, default=1)
@click.option('--batch-gpu', help='Samples per pass, adapt to fit on GPU', type=int, default=8)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
def generate_samplesheet(
    network_pkl: str,
    truncation_psi: float,
    seed: int,
    max_classes: int,
    samples_per_class: int,
    classes_per_row: int,
    batch_gpu: int,
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

    # handle unconditional models
    if not G.c_dim:
        G.c_dim = 1
        max_classes = 1
        n_total = samples_per_class
        classes_per_row = 1
        grid_width = min(32, samples_per_class)
    else:
        max_classes = min(max_classes, G.c_dim)
        n_total = max_classes*samples_per_class
        grid_width = samples_per_class * classes_per_row

    # allocate inputs
    labels = torch.repeat_interleave(F.one_hot(torch.arange(0, G.c_dim)), repeats=samples_per_class, dim=0)
    labels = labels[:n_total]

    grid_z = torch.from_numpy(np.random.RandomState(seed).randn(samples_per_class, G.z_dim)).repeat(max_classes, 1)
    grid_z = grid_z.to(device).split(batch_gpu)
    grid_c = labels.to(device).split(batch_gpu)

    # generate samples and save to disc
    images = []
    for z, c in tqdm(zip(grid_z, grid_c), total=len(grid_z)):
        images.append(G(z=z, c=c, truncation_psi=truncation_psi).cpu())

    # calculate grid size, pad if necessary
    grid_height = int(np.ceil(n_total / grid_width))
    grid_size = grid_width, grid_height
    padding_batchsize = np.prod(grid_size) - n_total
    images += [torch.zeros(images[0][0].shape).repeat(padding_batchsize, 1, 1, 1)]

    save_image_grid(
        img=torch.cat(images).numpy(),
        fname=run_dir / 'sheet.png',
        drange=[-1, 1],
        grid_size=grid_size
    )

if __name__ == "__main__":
    generate_samplesheet()
