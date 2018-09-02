import os
def make_directory(conf_data):
        """Making directory to save results and trained-model.
        
        Parameters
    ----------
        conf_data: dict
        Dictionary containing all parameters and objects.               

    Returns
    -------
    conf_data: dict
        Dictionary containing all parameters and objects.       

    """

        if not os.path.exists(conf_data['result_path']):
            os.makedirs(conf_data['result_path'])
        if not os.path.exists(conf_data['save_model_path']):
            os.makedirs(conf_data['save_model_path'])
        if not os.path.exists(conf_data['performance_log']):
            os.makedirs(conf_data['performance_log'])
        if not os.path.exists(conf_data['save_model_path']+'/Seq'):
            os.makedirs(conf_data['save_model_path']+'/Seq')
        if not os.path.exists(conf_data['save_model_path']+'/Seq'):
            os.makedirs(conf_data['save_model_path']+'/Seq')
        return conf_data
