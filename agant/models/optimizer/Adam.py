import torch

class optimizer_block:
	def __init__(self):
		super(optimizer_block, self).__init__()
		
	def optimizer(self,net,learning_rate,b1,b2):
		return torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(b1, b2))