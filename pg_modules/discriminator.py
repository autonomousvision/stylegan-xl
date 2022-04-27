import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import pickle

from training.diffaug import DiffAugment
from training.networks_stylegan2 import FullyConnectedLayer
from pg_modules.blocks import conv2d, DownBlock, DownBlockPatch
from pg_modules.projector import F_RandomProj
from feature_networks.constants import VITS

class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False):
        super().__init__()

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)

class SingleDiscCond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False, c_dim=1000, cmap_dim=64, rand_embedding=False):
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)

        # Pretrained Embeddings
        embed_path = 'in_embeddings/tf_efficientnet_lite0.pkl'
        with open(embed_path, 'rb') as f:
            self.embed = pickle.Unpickler(f).load()['embed']
        print(f'loaded imagenet embeddings from {embed_path}: {self.embed}')
        if rand_embedding:
            self.embed.__init__(num_embeddings=self.embed.num_embeddings, embedding_dim=self.embed.embedding_dim)
            print(f'initialized embeddings with random weights')

        self.embed_proj = FullyConnectedLayer(self.embed.embedding_dim, self.cmap_dim, activation='lrelu')

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out

class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4, 5]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, patch=patch)],

        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c, rec=False):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits

class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        backbones,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.backbones = backbones
        self.diffaug = diffaug
        self.interp224 = interp224

        # get backbones and multi-scale discs
        feature_networks, discriminators = [], []

        for i, bb_name in enumerate(backbones):

            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(
                channels=feat.CHANNELS,
                resolutions=feat.RESOLUTIONS,
                **backbone_kwargs,
            )

            feature_networks.append([bb_name, feat])
            discriminators.append([bb_name, disc])

        self.feature_networks = nn.ModuleDict(feature_networks)
        self.discriminators = nn.ModuleDict(discriminators)

    def train(self, mode=True):
        self.feature_networks = self.feature_networks.train(False)
        self.discriminators = self.discriminators.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c):
        logits = []

        for bb_name, feat in self.feature_networks.items():

            # apply augmentation (x in [-1, 1])
            x_aug = DiffAugment(x, policy='color,translation,cutout') if self.diffaug else x

            # transform to [0,1]
            x_aug = x_aug.add(1).div(2)

            # apply F-specific normalization
            x_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(x_aug)

            # upsample if smaller, downsample if larger + VIT
            if self.interp224 or bb_name in VITS:
                x_n = F.interpolate(x_n, 224, mode='bilinear', align_corners=False)

            # forward pass
            features = feat(x_n)
            logits += self.discriminators[bb_name](features, c)

        return logits
