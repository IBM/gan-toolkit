import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    Implementation of CGAN discrminator is learnt from https://github.com/ZiJianZhao/SeqGAN-PyTorch.git
    A CNN for text classification
    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self,conf_data):
        super(Discriminator, self).__init__()
        num_classes = 2
        vocab_size = conf_data['GAN_model']['vocab_size']
        emb_dim = conf_data['discriminator']['embedding_dim']

        filter_sizes = conf_data['discriminator']['filter_sizes']
        #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        num_filters = conf_data['discriminator']['num_filters']
        #[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        dropout = conf_data['discriminator']['dropout']

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)