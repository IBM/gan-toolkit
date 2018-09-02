#CGAN 
import torch.nn as nn
import numpy as np
import torch

class Discriminator(nn.Module):
    """
    Implementation of CGAN discrminator is learnt from https://github.com/eriklindernoren/PyTorch-GAN.git
    """
    def __init__(self,conf_data):
        super(Discriminator, self).__init__()
        self.img_shape = (conf_data['discriminator']['channels'],conf_data['discriminator']['input_shape'],conf_data['discriminator']['input_shape'])
        self.n_classes = conf_data['GAN_model']['classes']
        self.label_embedding = nn.Embedding(self.n_classes, self.n_classes)

        self.model = nn.Sequential(
            nn.Linear(self.n_classes + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input

        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity