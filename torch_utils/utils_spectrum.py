import torch
from torch.fft import fftn


def roll_quadrants(data, backwards=False):
    """
    Shift low frequencies to the center of fourier transform, i.e. [-N/2, ..., +N/2] -> [0, ..., N-1]
    Args:
        data: fourier transform, (NxHxW)
        backwards: bool, if True shift high frequencies back to center

    Returns:
    Shifted fourier transform.
    """
    dim = data.ndim - 1

    if dim != 2:
        raise AttributeError(f'Data must be 2d but it is {dim}d.')
    if any(s % 2 == 0 for s in data.shape[1:]):
        raise RuntimeWarning('Roll quadrants for 2d input should only be used with uneven spatial sizes.')

    # for each dimension swap left and right half
    dims = tuple(range(1, dim+1))          # add one for batch dimension
    shifts = torch.tensor(data.shape[1:]) // 2 #.div(2, rounding_mode='floor')                # N/2 if N even, (N-1)/2 if N odd
    if backwards:
        shifts *= -1
    return data.roll(shifts.tolist(), dims=dims)


def batch_fft(data, normalize=False):
    """
    Compute fourier transform of batch.
    Args:
        data: input tensor, (NxHxW)

    Returns:
    Batch fourier transform of input data.
    """

    dim = data.ndim - 1     # subtract one for batch dimension
    if dim != 2:
        raise AttributeError(f'Data must be 2d but it is {dim}d.')
        
    dims = tuple(range(1, dim + 1))  # add one for batch dimension
    if normalize:
        norm = 'ortho'
    else:
        norm = 'backward'

    if not torch.is_complex(data):
        data = torch.complex(data, torch.zeros_like(data))
    freq = fftn(data, dim=dims, norm=norm)

    return freq


def azimuthal_average(image, center=None):
    # modified to tensor inputs from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """
    Calculate the azimuthally averaged radial profile.
    Requires low frequencies to be at the center of the image.
    Args:
        image: Batch of 2D images, NxHxW
        center: The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    Returns:
    Azimuthal average over the image around the center
    """
    # Check input shapes
    assert center is None or (len(center) == 2), f'Center has to be None or len(center)=2 ' \
                                                 f'(but it is len(center)={len(center)}.'
    # Calculate the indices from the image
    H, W = image.shape[-2:]
    h, w = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    if center is None:
        center = torch.tensor([(w.max() - w.min()) / 2.0, (h.max() - h.min()) / 2.0])

    # Compute radius for each pixel wrt center
    r = torch.stack([w-center[0], h-center[1]]).norm(2, 0)

    # Get sorted radii
    r_sorted, ind = r.flatten().sort()
    i_sorted = image.flatten(-2, -1)[..., ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.long()             # attribute to the smaller integer

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented, computes bin change between subsequent radii
    rind = torch.where(deltar)[0]  # location of changed radius

    # compute number of elements in each bin
    nind = rind + 1         # number of elements = idx + 1
    nind = torch.cat([torch.tensor([0]), nind, torch.tensor([H*W])])        # add borders
    nr = nind[1:] - nind[:-1]  # number of radius bin, i.e. counter for bins belonging to each radius

    # Cumulative sum to figure out sums for each radius bin
    if H % 2 == 0:
        raise NotImplementedError('Not sure if implementation correct, please check')
        rind = torch.cat([torch.tensor([0]), rind, torch.tensor([H * W - 1])])  # add borders
    else:
        rind = torch.cat([rind, torch.tensor([H * W - 1])])  # add borders
    csim = i_sorted.cumsum(-1, dtype=torch.float64)             # integrate over all values with smaller radius
    tbin = csim[..., rind[1:]] - csim[..., rind[:-1]]
    # add mean
    tbin = torch.cat([csim[:, 0:1], tbin], 1)

    radial_prof = tbin / nr.to(tbin.device)         # normalize by counted bins

    return radial_prof


def get_spectrum(data, normalize=False):
    dim = data.ndim - 1  # subtract one for batch dimension
    if dim != 2:
        raise AttributeError(f'Data must be 2d but it is {dim}d.')

    freq = batch_fft(data, normalize=normalize)
    power_spec = freq.real ** 2 + freq.imag ** 2
    N = data.shape[1]
    if N % 2 == 0:      # duplicate value for N/2 so it is put at the end of the spectrum
                        # and is not averaged with the mean value
        N_2 = N//2
        power_spec = torch.cat([power_spec[:, :N_2+1], power_spec[:, N_2:N_2+1], power_spec[:, N_2+1:]], dim=1)
        power_spec = torch.cat([power_spec[:, :, :N_2+1], power_spec[:, :, N_2:N_2+1], power_spec[:, :, N_2+1:]], dim=2)

    power_spec = roll_quadrants(power_spec)
    power_spec = azimuthal_average(power_spec)
    return power_spec


def plot_std(mean, std, x=None, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1)

    # plot error margins in same color as line
    err_kwargs = {
        'alpha': 0.3
    }

    if 'c' in kwargs.keys():
        err_kwargs['color'] = kwargs['c']
    elif 'color' in kwargs.keys():
        err_kwargs['color'] = kwargs['color']

    if x is None:
        x = torch.linspace(0, 1, len(mean))     # use normalized x axis
    ax.plot(x, mean, **kwargs)
    ax.fill_between(x, mean-std, mean+std, **err_kwargs)

    return ax
