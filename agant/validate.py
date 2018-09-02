from validator import Required, Not, Truthy, Blank, Range, Equals, In, validate
def validate_check(conf_data):
	"""Validate configuration file.
	
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
	TODO:
	- Add more checks for correctness of values in the config file.
	- Throw appropriate error.
	- Change to try-catch.

	- Check '' or None for datapath
	"""

	rules_generator = {
		"choice":[Required,In(["gan", "cgan", "wgan","wgan_gp","dcgan","custom","seq_gan"])]
	}

	rules_discriminator = {
		"choice":[Required,In(["gan", "cgan", "wgan","wgan_gp","dcgan","custom","seq_gan"])]
	}

	rules_model = {

	}

	rules = {
		"data_path":[Required, Truthy()]
	}

	validation_1 = validate(rules_generator, conf_data['generator'])
	validation_2 = validate(rules_discriminator,conf_data['discriminator'])
	validation_3 = validate(rules,conf_data)
	#validation_4 = validate(rules_model,conf_data['GAN_model'])

	#if (validation_1[0] and validation_2[0] and validation_3[0] and validation_4[0]) == False:
	if (validation_1[0] and validation_2[0] and validation_3[0]) == False:
		print ("Errors")
		print("------")
		if validation_1[0] is False:
			print ("Generator --> ")
			print (validation_1[1])
		if validation_2[0] is False:
			print ("Discriminator --> ")
			print (validation_2[1])
		if validation_3[0] is False:
			print ("Others --> ")
			print (validation_3[1])
		# if validation_4[0] is False:
		# 	print ("Model --> ")
		# 	print (validation_4[1])
		exit()