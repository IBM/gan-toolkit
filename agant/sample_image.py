from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np

def sample_image(n_row, batches_done, conf_data):
	"""Saving images for CGAN. Grid of generated images ranging through all n-classes.
	
	Parameters
    ----------
    n_row: int
    	Number of rows of samples.

    batches_done: int 
    	Count of number of batches from dataloader seen.

	conf_data: dict
    	Dictionary containing all parameters and objects. 		

	"""

	generator = conf_data['generator_model']
	g_latent_dim = int(conf_data['generator']['latent_dim'])
	UseTensor = conf_data['Tensor']
	z = Variable(UseTensor(conf_data['Tensor'](np.random.normal(0, 1, (n_row**2, g_latent_dim))))) # Sample noise
	# Get labels ranging from 0 to n_classes for n rows
	labels = np.array([num for _ in range(n_row) for num in range(n_row)])
	labels = Variable(conf_data['LongTensor'](labels))
	gen_imgs = generator(z, labels)
	save_image(gen_imgs.data[:25], conf_data['result_path']+'/%d.png' % batches_done, nrow=n_row, normalize=True)