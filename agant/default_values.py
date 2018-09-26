import json
import datetime
import os

def default_values(conf_data):
	"""Setting default values based on choice on networks.
	
	If different set of generator and discriminator are used. Defulat training process of the generator is loaded.
	Default values are configured in template_file.json .

	Parameters
    ----------
	conf_data: dict
    	Dictionary containing all parameters and objects. 		

    Returns
    -------
    conf_data: dict
    	Dictionary containing all parameters and objects. 		

	"""
	"""
	Search for this -- Added Here
	"""
	with open('template_file.json') as json_data_file:
		template_data = json.load(json_data_file)	

	generator_model_choice = conf_data['generator']['choice']
	discriminator_model_choice = conf_data['discriminator']['choice']

	generator_template = template_data[generator_model_choice]['generator']
	discriminator_template = template_data[discriminator_model_choice]['discriminator']
	GAN_model_template = template_data[generator_model_choice]['GAN_model']

	generator_parameter = generator_template.keys()
	discriminator_parameter = discriminator_template.keys()
	GAN_model_parameter = GAN_model_template.keys()


	for param in generator_parameter:
		if conf_data['generator'].get(param) == None or conf_data['generator'][param] == None:
			conf_data['generator'][param] = generator_template[param]
		else:
			if(isinstance(generator_template[param], dict)):
				for sub_param in generator_template[param].keys():
					if conf_data['generator'][param].get(sub_param) == None:
						conf_data['generator'][param][sub_param] = generator_template[param][sub_param]
	for param in discriminator_parameter:
		if conf_data['discriminator'].get(param) == None or conf_data['discriminator'][param] == None:
			conf_data['discriminator'][param] = discriminator_template[param]
		else:
			if(isinstance(discriminator_template[param], dict)):
				for sub_param in discriminator_template[param].keys():
					if conf_data['discriminator'][param].get(sub_param) == None:
						conf_data['discriminator'][param][sub_param] = discriminator_template[param][sub_param]
	if conf_data.get('GAN_model',None) == None:
		conf_data['GAN_model'] = {}

	
	for param in GAN_model_parameter:
		if conf_data['GAN_model'].get(param) == None or conf_data['GAN_model'][param] == None:
			conf_data['GAN_model'][param] = GAN_model_template[param]

	directory_name = conf_data['generator']['choice'] + "_" + conf_data['discriminator']['choice'] + "_" +datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
	if conf_data.get('save_model_path','') == '' or conf_data['save_model_path'] == None:
		os.makedirs(template_data['save_model_path'] + '/' + directory_name)
		conf_data['save_model_path'] = template_data['save_model_path'] + '/' + directory_name
	if conf_data.get('result_path','') == '' or conf_data['result_path'] == None:
		os.makedirs(template_data['result_path'] + '/' + directory_name)
		conf_data['result_path'] = template_data['result_path'] + '/' + directory_name
	if conf_data.get('performance_log','') == '' or conf_data['performance_log'] == None:
		os.makedirs(template_data['performance_log'] + '/' + directory_name)
		conf_data['performance_log'] =  template_data['performance_log'] + '/' + directory_name
	if conf_data.get("sample_interval",0) == 0 or conf_data["sample_interval"] == None:
		conf_data["sample_interval"] = template_data["sample_interval"]
	if conf_data.get('metric_evaluate','') == '' or conf_data['metric_evaluate'] == None:
		conf_data['metric_evaluate'] = template_data['metric_evaluate']
	
	return conf_data