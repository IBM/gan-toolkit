import torch

def save_models(conf_data):
	""" Saving the trained discriminator and generator.
	
	Parameters
    ----------
	conf_data: dict
    	Dictionary containing all parameters and objects. 		

    Returns
    -------
    conf_data: dict
    	Dictionary containing all parameters and objects. 		

	"""
	torch.save(conf_data['generator_model'].state_dict(),conf_data['save_model_path']+'/'+'generator.pt')
	#print (conf_data['save_model_path']+'/'+'generator.pt')
	torch.save(conf_data['discriminator_model'].state_dict(),conf_data['save_model_path']+'/'+'discriminator.pt')
	#print ("Saved")
	return conf_data