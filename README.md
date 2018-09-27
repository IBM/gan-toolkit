# gan-toolkit
The aim of the toolkit is to provide a highly flexible, no-code way of implementing GAN models. By providing the details of a GAN model, in an intuitive config file or as command line arguments, the code could be generated for training the GAN model. With very little or no prior knowledge about GAN, people could play around with different formulations of GAN and mix-and-match GAN modules to create novel models, as well.

The toolkit is still under development. Comments, bug reports, and extensions are highly appreciated. Get in touch with us on [Slack](https://gan-toolkit.slack.com) (invite [here](https://join.slack.com/t/gan-toolkit/shared_invite/enQtNDQzMzA4OTM5NTU0LTczZTM4MmYyNmE4ZjI0ZGU5MTdkM2MzMWZkNDFmMjNhNTBhMTE3MmI2YmY2YWQxMzkzODExNjBiYThjMzZiOTk))!

## Modular GAN Architecture

![GAN Architecture](images/gan_toolkit_architecture.png?raw=true "Modular GAN Architecture")

## Quick Start

0. (Optional) If you want to setup an anaconda environment

    a. Install Anaconda from [here](https://conda.io/docs/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages)

    b. Create a conda environment
    ```shell
    $ conda create -n gantoolkit python=3.6 anaconda
    ```

    c. Activate the conda environment
    ```shell
    $ source activate gantoolkit
    ```

1. Clone the code

    ```shell
    $ git clone https://github.com/IBM/gan-toolkit
    $ cd gan-toolkit
    ```

2. Install all the requirements. Tested for Python 3.5.x+

    ```shell
    $ pip install -r requirements.txt
    ```

3. Train the model using a configuration file. (Many samples are provided in the `configs` folder)

    ```shell
    $ cd agant
    $ python main.py --config configs/gan_gan.json
    ```

4. Default input and output paths (override thse paths in the config file)

    
    `logs/` : training logs

    `saved_models/` : saved trained models

    `train_results/` : saved all the intermediate generated images

    `datasets/` : input dataset path 

## Implemented GAN Models

1. Vanilla GAN: Generative Adversarial Learning ([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661))

2. C-GAN: Conditional Generative Adversarial Networks ([Mirza et al., 2014](https://arxiv.org/abs/1411.1784))

3. DC-GAN: Deep Convolutional Generative Adversarial Network  ([Radford et al., 2016](https://arxiv.org/abs/1511.06434))

4. W-GAN: Wasserstein GAN    ([Arjovsky et al., 2017](https://arxiv.org/abs/1701.07875))

5. W-GAN-GP: Improved Training of Wasserstein GANs  ([Goodfellow et al., 2017](https://arxiv.org/abs/1704.00028))


## Config File Structure and Details

The config file is a set of key-value pairs in JSON format. A collection of sample config files are provided [here](./agant/configs/)

The basic structure of the `config` json file is as follows,

```Javascript
    { 
        "generator":{
            "choice":"gan"
        },
        "discriminator":{
            "choice":"gan"
        },
        "data_path":"datasets/dataset1.p",
        "metric_evaluate":"MMD"
    }
```

The details of the config files are provided [here](https://github.com/IBM/gan-toolkit/wiki/Config-File-Structure-and-Details)

## Comparison with Other Toolkits

Realizing the importance of easiness in training GAN models, there are a few other toolkits available in open source domain such as [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN), [TF-GAN](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan/), [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN). However, our `gan-toolkit` has the following advantages:

 - Highly modularized representation of GAN model for easy mix-and-match of components across architectures. For instance, one can use the `generator` component from DCGAN and the `discriminator` component from CGAN, with the training process of WGAN.

  - An abstract representation of GAN architecture to provide multi-library support. Currently, we are providing a PyTorch support for the provided `config` file, while in future, we plan to support Keras and Tensorflow as well. Thus, the abstract representation is library agnostic.

  - Coding free way of designing GAN models. A simple JSON file is required to define a GAN architecture and there is no need for writing any training code to train the GAN model.

## TO-DO

Immediate tasks:
 - [ ]  Better the performance of seq-GAN 
 - [ ]  Implement a textGAN for text based applications
 - [ ]  Study and implement better transfer learning approaches
 - [ ]  Check out different weight init for GANs 
 - [ ]  Check if making optimizer as cuda is also important or not
 - [ ]  Check the input for generator and discriminator to conf_data
 - [ ]  Find a smart way to check the size of the reward
 
Long term tasks:
 - [ ]  Implement driver and support for Keras
  - [ ]  Implement driver and support for Tensorflow
 - [ ]  Implement more popular GAN models in this framework
 - [ ]  Implement more metrics to evaluate different GAN models
 - [ ]  Support multimodal data generation for GAN frameworks

## Credits

We would like to thank Raunak Sinha ([email](raunak15075@iiitd.ac.in)) who interned with us during summer 2018 and contributed heavily to this toolkit.
