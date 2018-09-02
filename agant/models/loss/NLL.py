import torch
import numpy as np
from torch.autograd import Variable

class loss_block:
    def __init__(self):
        super(loss_block, self).__init__()
        self.criterion = torch.nn.NLLLoss(size_average=False)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.criterion.cuda()
    def loss(self,input_vals,lables):
        return self.criterion(input_vals,lables)