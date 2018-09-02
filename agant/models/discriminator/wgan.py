#WGAN
import torch.nn as nn
import numpy as np
class Discriminator(nn.Module):
    """
    Implementation of WGAN discrminator is learnt from https://github.com/eriklindernoren/PyTorch-GAN.git
    """
    def __init__(self,conf_data):
        super(Discriminator, self).__init__()
        self.img_shape = (conf_data['discriminator']['channels'],conf_data['discriminator']['input_shape'],conf_data['discriminator']['input_shape'])

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
