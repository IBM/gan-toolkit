#WGAN
import torch.nn as nn
import numpy as np
class Generator(nn.Module):
    """
    Implementation of WGAN generator is learnt from https://github.com/eriklindernoren/PyTorch-GAN.git
    """
    def __init__(self,conf_data):
        super(Generator, self).__init__()
        self.img_shape = (conf_data['generator']['channels'],conf_data['generator']['input_shape'],conf_data['generator']['input_shape'])
        self.latent_dim = conf_data['generator']['latent_dim']

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img
