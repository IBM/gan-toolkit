import numpy as np
from torch.autograd import Variable
def evaluate(conf_data):
	"""Metric based evaluation of trained model.
	
	Parameters
    ----------
	conf_data: dict
    	Dictionary containing all parameters and objects. 		

    Returns
    -------
    score: int
    	Evaluation metric score. 		

	"""
	cuda = conf_data['cuda']
	evaluation_metric = conf_data['evaluation_metric']
	generator = conf_data['generator_model']
	g_latent_dim = int(conf_data['generator']['latent_dim'])
	Tensor = conf_data['Tensor']
	LongTensor = conf_data['LongTensor']

	true_data = conf_data['transformed']
	true_data_numpy = []
	for iterator in true_data:
		if conf_data['GAN_model']['data_label'] == 1:
			i,_ = iterator
		else:
			i = iterator
		t = np.asarray(i)
		true_data_numpy.append(t)
	true_data_numpy = np.array(true_data_numpy)
	true_data = Variable(Tensor(true_data_numpy))

	z = Variable(Tensor(np.random.normal(0, 1, (true_data.size(0), g_latent_dim))))
	if conf_data['GAN_model']['data_label'] == 1:
		gen_labels = Variable(LongTensor(np.random.randint(0, conf_data['GAN_model']['classes'], true_data.shape[0])))
		samples = generator(z,gen_labels)	
	else:
		samples = generator(z)
	
	#generated_data = samples.data.numpy()
	score = evaluation_metric.calculate(true_data,samples)
	if cuda:
		score = score.cpu().data.numpy()
	else:
		score = score.data.numpy()          	
	log_file = conf_data['log_file']
	log_file.write(conf_data['metric_evaluate']+" score of the trained GAN is = {}\n".format(score))
	#log_file.close()
	return score
