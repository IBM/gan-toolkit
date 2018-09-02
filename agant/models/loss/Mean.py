import torch
import numpy as np
from torch.autograd import Variable

class loss_block:
	def __init__(self):
		super(loss_block, self).__init__()
	def loss(self,input_vals,lables):
		return torch.mean(input_vals)