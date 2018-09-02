import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd

def compute_gradient_penalty(conf_data):
	"""Calculate gradient penalty for WGAN-GP.
	
	Parameters
    ----------
	conf_data: dict
    	Dictionary containing all parameters and objects. 		

    Returns
    -------
    conf_data: dict
    	Dictionary containing all parameters and objects. 		

	"""
	D = conf_data['discriminator_model']
	real_samples = conf_data['real_data_sample']
	fake_samples = conf_data['fake_data_sample']
	# Random weight term for interpolation between real and fake samples
	alpha = conf_data['Tensor'](np.random.random((real_samples.size(0), 1, 1, 1)))

	# Get random interpolation between real and fake samples
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	d_interpolates = D(interpolates)
	fake = Variable(conf_data['Tensor'](real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
	# Get gradient w.r.t. interpolates
	gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
							  grad_outputs=fake, create_graph=True, retain_graph=True,
							  only_inputs=True)[0]
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

	conf_data['gradient_penalty'] = gradient_penalty
	return conf_data