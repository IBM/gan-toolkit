# gan-toolkit
The aim of the toolkit is to provide a highly flexible, no-code way of implementing GAN models. By providing the details of a GAN model, in an intuitive config file or as command line arguments, the code could be generated for training the GAN model. With very little or no prior knowledge about GAN, people could play around with different formulations of GAN and mix-and-match GAN modules to create novel models, as well.

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

4. Cycle-GAN: Cycle-Consistent Adversarial Networks ([Zhu et al., 2017](https://arxiv.org/abs/1703.10593))

5. W-GAN: Wasserstein GAN    ([Arjovsky et al., 2017](https://arxiv.org/abs/1701.07875))

6. W-GAN-GP: Improved Training of Wasserstein GANs  ([Goodfellow et al., 2017](https://arxiv.org/abs/1704.00028))


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

The details of the config files are provided here:

- `generator`: < json > value which contains the details of the generator module. The available parameters and possible values are:
    - `choice`: ["gan", "cgan", "dcgan", "cycle_gan", "wgan", "wgan_gp"] // choice of the generator module
    - `input_shape`: < int > // row size of the input image
    - `channels`: < int > // number of channels in the input image
    - `latent_dim`: < int > // the size of the input random vector
    - `input`: "[(g_channels, g_input_shape, g_input_shape), g_latent_dim]" // of the given format of input data
    - `loss`: ["Mean", "MSE", "BCE", "NLL"] // choice of the loss function
    - `optimizer`: < json > value of the optimizer and it's parameters
        - `choice`: ["Adam", "RMSprop"]
        - `learning_rate`: < int > // learning rate of the optimizer
        - `b1`: < int > //  Coefficients used for computing running averages of gradient and its square. Used in Adam optimizer.
        - `b2`: < int > //  Coefficients used for computing running averages of gradient and its square. Used in Adam optimizer.

- `generator`: < json > value which contains the details of the discriminator module. The available parameters and possible values are:
    - `choice`: ["gan", "cgan", "dcgan", "cycle_gan", "wgan", "wgan_gp", "seq_gan"] // choice of the discriminator module
    - `input_shape`: < int > // row size of the input image
    - `channels`: < int > // number of channels in the input image
    - `input`: "[(g_channels, g_input_shape, g_input_shape), g_latent_dim]" // of the given format of input data
    - `loss`: ["Mean", "MSE", "BCE", "NLL"] // choice of the loss function
    - `optimizer`: < json > value of the optimizer and it's parameters
        - `choice`: ["Adam", "RMSprop"]
        - `learning_rate`: < int > // learning rate of the optimizer
        - `b1`: < int > //  Coefficients used for computing running averages of gradient and its square. Used in Adam optimizer.
        - `b2`: < int > //  Coefficients used for computing running averages of gradient and its square. Used in Adam optimizer.

- `data_path`: "path/of/data/in/local/system"

- `metric_evaluate`: ["MMD", "FID"]  // maximum mean discrepancy

- `GAN_model`: < json > format providing the meta details for training the GAN model
    - `epochs`: < int > // number of epochs for training
    - `mini_batch_size`: < int > // size of each mini batch
    - `clip_value`: < int > // the peak clip value
    - `n_critic`: < int > // the number of critics required for wgan
    - `lambda_gp`: < int > // the parameter for wgan_gp
    - `data_label`: < int > // the parameter required for cgan
    - `classes`: < int > // the number of classes in the given real data
    - `seq`: < binary > // 0 or 1 on whether the generation is single value or sequential. Used for seq_gan

- `result_path`: "path/to/write/resulting/images" 

- `save_model_path`:  "path/to/write/trained/model" 

- `performance_log`:  "path/to/write/training/logs" 

- `sample_interval`:  "frequency/to/write/resulting/images"

## Comparison with Other Toolkits

Realizing the importance of easiness in training GAN models, there are a few other toolkits available in open source domain such as [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN), [TF-GAN](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan/), [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN). However, our `gan-toolkit` has the following advantages:

 - Highly modularized representation of GAN model for easy mix-and-match of components across architectures. For instance, one can use the `generator` component from DCGAN and the `discriminator` component from CGAN, with the training process of WGAN.

  - An abstract representation of GAN architecture to provide multi-library support. Currently, we are providing a PyTorch support for the provided `config` file, while in future, we plan to support Keras and Tensorflow as well. Thus, the abstract representation is library agnostic.

  - Coding free way of designing GAN models. A simple JSON file is required to define a GAN architecture and there is no need for writing any training code to train the GAN model.

## TO-DO

 - Better the performance of seq-GAN 
 - Implement a textGAN for text based applications
 - Study and implement better transfer learning approaches
 - Check out different weight init for GANs 
 - Check if making optimizer as cuda is also important or not
 - Check the input for generator and discriminator to conf_data
 - Find a smart way to check the size of the reward

## Credits

We would like to thank Raunak Sinha ([email](raunak15075@iiitd.ac.in)) who interned with us during summer 2018 and contributed heavily to this toolkit.