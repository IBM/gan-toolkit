import argparse
import json

def argument_parser():
	"""Argument Parser Fucntion.
	
	Parameters
    ----------

    Returns
    -------
    conf_data: dict
    	Dictionary containing all parameters and objects.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--configuration', type=str, help='Model configuration file')
	parser.add_argument('--result_path', type=str, help='Path to save results at')
	parser.add_argument('--save_model_path', type=str, help='Path to save model at')
	parser.add_argument('--cuda', type=bool, help='Choice for selecting use of GPU')
	parser.add_argument('--performance_log', type=str, help='Path to file to record the model log while training and evauation')
	parser.add_argument('--epochs', type=int, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, help='Size of batch')
	parser.add_argument('--clip_value', type=int, help='Value for gragient clipping')
	parser.add_argument('--critic', type=int, help='Value of critic')
	parser.add_argument('--lambda_gp', type=int, help='Value of lambda for gradient penalty')
	parser.add_argument('--w_loss', type=int, help='Choosing using Wasserstein loss training process')
	parser.add_argument('--data_label', type=int, help='Whether data has lable values or not')
	parser.add_argument('--classes', type=int, help='Number of classes in data. Determine the use CGAN training process')
	parser.add_argument('--data_path', type=str, help='Path to data')
	parser.add_argument('--metric_evaluate', type=str, help='Choice of evalution metric')
	parser.add_argument('--sample_interval', type=int, help='Interval to sample during training')

	parser.add_argument('--g_choice', type=str, help='Choice of Generator')
	parser.add_argument('--g_pre_trained_path', type=str, help='Path to the pre-trained generator network')
	parser.add_argument('--g_input_shape', type=int, help='Input shape to the generator network')
	parser.add_argument('--g_latent_dim', type=int, help='Size of the noise vector')
	parser.add_argument('--g_channels', type=int, help='Number of channels in input image')
	parser.add_argument('--g_optimizer', type=str, help='Choice of optimizer for generator')
	parser.add_argument('--g_learning_rate', type=float, help='Value of learning rate for generator')
	parser.add_argument('--g_b1', type=float, help='Value of b1 for generator')
	parser.add_argument('--g_b2', type=float, help='Value of b2 for generator')
	parser.add_argument('--g_loss', type=str, help='Choice of loss for generator')

	parser.add_argument('--d_choice', type=str, help='Choice of Discriminator')
	parser.add_argument('--d_pre_trained_path', type=str, help='Path to the pre-trained discriminator network')
	parser.add_argument('--d_input_shape', type=int, help='Input shape to the discriminator network')
	parser.add_argument('--d_channels', type=int, help='Number of channels in input image')
	parser.add_argument('--d_optimizer', type=str, help='Choice of optimizer for discriminator')
	parser.add_argument('--d_learning_rate', type=float, help='Value of learning rate for discriminator')
	parser.add_argument('--d_b1', type=float, help='Value of b1 for discriminator')
	parser.add_argument('--d_b2', type=float, help='Value of b2 for discriminator')
	parser.add_argument('--d_loss', type=str, help='Choice of loss for discriminator')
	opt = parser.parse_args()
	
	if opt.configuration == None:
		conf_data = {}
		conf_data['GAN_model'] = {}
		conf_data['generator'] = {}
		conf_data['generator']['optimizer'] = {}
		conf_data['discriminator'] = {}
		conf_data['discriminator']['optimizer'] = {}

		conf_data['sample_interval'] = opt.sample_interval
		conf_data['result_path'] = opt.result_path
		conf_data['save_model_path'] = opt.save_model_path
		conf_data['cuda'] = opt.cuda
		conf_data['performance_log'] = opt.performance_log
		conf_data['data_path'] = opt.data_path
		conf_data['metric_evaluate'] = opt.metric_evaluate

		conf_data['GAN_model']['epochs'] = opt.epochs
		conf_data['GAN_model']['mini_batch_size'] = opt.batch_size
		conf_data['GAN_model']['clip_value'] = opt.clip_value
		conf_data['GAN_model']['n_critic'] = opt.critic
		conf_data['GAN_model']['lambda_gp'] = opt.lambda_gp
		conf_data['GAN_model']['w_loss'] = opt.w_loss
		conf_data['GAN_model']['data_label'] = opt.data_label
		conf_data['GAN_model']['classes'] = opt.classes

		conf_data['generator']['choice'] = opt.g_choice
		conf_data['generator']['pre_trained_path'] = opt.g_pre_trained_path
		conf_data['generator']['input_shape'] = opt.g_input_shape
		conf_data['generator']['latent_dim'] = opt.g_latent_dim
		conf_data['generator']['channels'] = opt.g_channels
		conf_data['generator']['optimizer']['choice'] = opt.g_optimizer
		conf_data['generator']['optimizer']['learning_rate'] = opt.g_learning_rate
		conf_data['generator']['optimizer']['b1'] = opt.g_b1
		conf_data['generator']['optimizer']['b2'] = opt.g_b2
		conf_data['generator']['loss'] = opt.g_loss

		conf_data['discriminator']['choice'] = opt.g_choice
		conf_data['discriminator']['pre_trained_path'] = opt.g_pre_trained_path
		conf_data['discriminator']['input_shape'] = opt.g_input_shape
		conf_data['discriminator']['channels'] = opt.g_channels
		conf_data['discriminator']['optimizer']['choice'] = opt.g_optimizer
		conf_data['discriminator']['optimizer']['learning_rate'] = opt.g_learning_rate
		conf_data['discriminator']['optimizer']['b1'] = opt.g_b1
		conf_data['discriminator']['optimizer']['b2'] = opt.g_b2
		conf_data['discriminator']['loss'] = opt.g_loss
	else: 
		config_file = opt.configuration
		with open(config_file) as json_data_file:
			conf_data = json.load(json_data_file)
	return conf_data