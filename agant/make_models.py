import importlib.util
import torch

def make_models(conf_data):
	"""Makes models from configuration parameter. Instantiate networks, loss and optimizers for both generator and discriminator. 
	
	Parameters
    ----------
	conf_data: dict
    	Dictionary containing all parameters and objects. 		

    Returns
    -------
    conf_data: dict
    	Dictionary containing all parameters and objects. 		

	"""
	generator_model_choice = conf_data['generator']['choice']
	discriminator_model_choice = conf_data['discriminator']['choice']

	generator_loss_choice = conf_data['generator']['loss']
	discriminator_loss_choice = conf_data['discriminator']['loss']

	generator_optimizer_choice =  conf_data['generator']['optimizer']['choice']
	discriminator_optimizer_choice = conf_data['discriminator']['optimizer']['choice']

	Generator = getattr(importlib.import_module('models.generator.'+generator_model_choice),'Generator')
	Discriminator = getattr(importlib.import_module('models.discriminator.'+discriminator_model_choice),'Discriminator')

	g_loss_block = getattr(importlib.import_module('models.loss.'+generator_loss_choice),'loss_block')
	d_loss_block = getattr(importlib.import_module('models.loss.'+discriminator_loss_choice),'loss_block')
	g_optimizer_block = getattr(importlib.import_module('models.optimizer.'+generator_optimizer_choice),'optimizer_block')
	
	d_optimizer_block = getattr(importlib.import_module('models.optimizer.'+discriminator_optimizer_choice),'optimizer_block')

	evaluation_metric= getattr(importlib.import_module('evaluation.'+conf_data['metric_evaluate']),'Metric')

	#Commented out
	# g_channels = int(conf_data['generator']['channels'])
	# g_input_shape = int(conf_data['generator']['input_shape'])
	# g_latent_dim = int(conf_data['generator']['latent_dim'])

	# """Added Here"""
	# g_embedding_dim =int(conf_data['generator']['embedding_dim'])
	# g_hidden_dim = int(conf_data['generator']['hidden_dim'])
	# g_sequece_length = int(conf_data['generator']['sequece_length'])
	# vocab_size = int(conf_data['GAN_model']['vocab_size'])

	# d_channels = int(conf_data['discriminator']['channels'])
	# d_input_shape = int(conf_data['discriminator']['input_shape'])

	# classes = int(conf_data['GAN_model']['classes'])

	# g_net_inp = conf_data['g_input']
	# g_net_inp = eval(g_net_inp)

	# d_net_inp = conf_data['d_input']
	# d_net_inp = eval(d_net_inp)

	#generator = Generator(g_net_inp)
	generator = Generator(conf_data)
	discriminator = Discriminator(conf_data)

	if conf_data['generator'].get('pre_trained_path','') != '' and conf_data['generator']['pre_trained_path'] != None:
		generator.load_state_dict(torch.load(conf_data['generator']['pre_trained_path']))

	if conf_data['discriminator'].get('pre_trained_path','') != '' and conf_data['discriminator']['pre_trained_path'] != None:
		discriminator.load_state_dict(torch.load(conf_data['discriminator']['pre_trained_path']))
	# print ("------------------------------------------------------------------------------------------------")
	# print ("Generator")

	# for p in generator.parameters():
	# 	print (p.size())
	# print ("------------------------------------------------------------------------------------------------")
	# print ("Discriminator")
	# for p in discriminator.parameters():
	# 	print (p.size())
	# print ("------------------------------------------------------------------------------------------------")
	# exit()
	if conf_data['cuda'] == True:
		generator.cuda()
		discriminator.cuda()

	g_loss_func = g_loss_block()
	d_loss_func = d_loss_block()

	g_learning_rate = float(conf_data['generator']['optimizer']['learning_rate'])
	g_b1 = float(conf_data['generator']['optimizer']['b1'])
	g_b2 = float(conf_data['generator']['optimizer']['b2'])

	d_learning_rate = float(conf_data['discriminator']['optimizer']['learning_rate'])
	d_b1 = float(conf_data['discriminator']['optimizer']['b1'])
	d_b2 = float(conf_data['discriminator']['optimizer']['b2'])

	g_optimizer_func = g_optimizer_block()
	d_optimizer_func = d_optimizer_block()
	optimizer_G = g_optimizer_func.optimizer(generator,g_learning_rate,g_b1,g_b2) 
	optimizer_D = d_optimizer_func.optimizer(discriminator,d_learning_rate,d_b1,d_b2)
	
	conf_data['generator_model'] = generator
	conf_data['discriminator_model'] = discriminator
	conf_data['generator_optimizer'] = optimizer_G
	conf_data['discriminator_optimizer'] = optimizer_D
	conf_data['generator_loss'] = g_loss_func
	conf_data['discriminator_loss'] = d_loss_func
	conf_data['evaluation_metric'] = evaluation_metric() 

	return conf_data