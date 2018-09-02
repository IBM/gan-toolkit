import torch
def cuda_check(conf_data):
	"""Checking run environment for cuda enabled GPU.
	
	Parameters
    ----------
	conf_data: dict
    	Dictionary containing all parameters and objects. 		

    Returns
    -------
    conf_data: dict
    	Dictionary containing all parameters and objects. 		

	"""
	if conf_data.get('cuda',None) == None or conf_data['cuda'] == None:
		cuda = True if torch.cuda.is_available() else False
		conf_data['cuda'] = cuda
	elif conf_data['cuda'] == 'True':
		conf_data['cuda'] = True
	elif conf_data['cuda'] == 'False':
		conf_data['cuda'] = False
	return conf_data
