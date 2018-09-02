import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from training_fucntion_generator import training_fucntion_generator
from compute_gradient_penalty import compute_gradient_penalty
from sample_image import sample_image

#See how many of these can be removed
from data_iter import GenDataIter, DisDataIter
from target_lstm import TargetLSTM
from rollout import Rollout
import torch.nn as nn
from sequential_generate_sample import generate_samples
from sequential_eval_epoch import eval_epoch
from sequential_train_epoch import train_epoch

def train_GAN(conf_data):

    """Training Process for GAN.
    
    Parameters
    ----------
    conf_data: dict
        Dictionary containing all parameters and objects.       

    Returns
    -------
    conf_data: dict
        Dictionary containing all parameters and objects.       

    """
    seq = conf_data['GAN_model']['seq']
    if seq == 1:
        pre_epoch_num = conf_data['generator']['pre_epoch_num']
        GENERATED_NUM = 10000 
        EVAL_FILE = 'eval.data'
        POSITIVE_FILE = 'real.data'
        NEGATIVE_FILE = 'gene.data'
    temp = 1 #TODO Determines how many times is the discriminator updated. Take this as a value input
    epochs = int(conf_data['GAN_model']['epochs'])
    if seq == 0:
        dataloader = conf_data['data_learn']
    mini_batch_size = int(conf_data['GAN_model']['mini_batch_size'])
    data_label = int(conf_data['GAN_model']['data_label'])
    cuda = conf_data['cuda']
    g_latent_dim = int(conf_data['generator']['latent_dim'])
    classes = int(conf_data['GAN_model']['classes'])

    w_loss = int(conf_data['GAN_model']['w_loss'])

    clip_value = float(conf_data['GAN_model']['clip_value'])
    n_critic = int(conf_data['GAN_model']['n_critic'])

    lambda_gp = int(conf_data['GAN_model']['lambda_gp'])

    log_file = open(conf_data['performance_log']+"/log.txt","w+")
    #Covert these to parameters of the config data
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    conf_data['Tensor'] = Tensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    conf_data['LongTensor'] = LongTensor
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    conf_data['FloatTensor'] = FloatTensor

    conf_data['epochs'] = epochs

    #print ("Just before training")
    if seq == 1: #TODO: Change back to 1 
        target_lstm = TargetLSTM(conf_data['GAN_model']['vocab_size'], conf_data['generator']['embedding_dim'], conf_data['generator']['hidden_dim'], conf_data['cuda'])
        if cuda == True:
            target_lstm = target_lstm.cuda()
        conf_data['target_lstm'] = target_lstm
        gen_data_iter = GenDataIter('real.data', mini_batch_size)
        generator = conf_data['generator_model']
        discriminator = conf_data['discriminator_model']
        g_loss_func = conf_data['generator_loss']
        d_loss_func = conf_data['discriminator_loss']
        optimizer_D = conf_data['discriminator_optimizer']
        optimizer_G = conf_data['generator_optimizer']
        #print('Pretrain with MLE ...')
        for epoch in range(pre_epoch_num): #TODO: Change the range
            loss = train_epoch(generator, gen_data_iter, g_loss_func, optimizer_G,conf_data,'g')
            print('Epoch [%d] Model Loss: %f'% (epoch, loss))
            generate_samples(generator, mini_batch_size, GENERATED_NUM, EVAL_FILE,conf_data)
            eval_iter = GenDataIter(EVAL_FILE, mini_batch_size)
            loss = eval_epoch(target_lstm, eval_iter, g_loss_func,conf_data)
            print('Epoch [%d] True Loss: %f' % (epoch, loss))

        dis_criterion = d_loss_func
        dis_optimizer = optimizer_D
        #TODO: Understand why the below two code line were there ?        
        # if conf_data['cuda']:
        #     dis_criterion = dis_criterion.cuda()

        #print('Pretrain Dsicriminator ...')
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, mini_batch_size)
        for epoch in range(5): #TODO: change back 5
            generate_samples(generator, mini_batch_size, GENERATED_NUM, NEGATIVE_FILE,conf_data)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, mini_batch_size)
            for _ in range(3): #TODO: change back 3
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer,conf_data,'d')
                print('Epoch [%d], loss: %f' % (epoch, loss))
        conf_data['generator_model'] = generator
        conf_data['discriminator_model'] = discriminator
        torch.save(conf_data['generator_model'].state_dict(),conf_data['save_model_path']+'/Seq/'+'pre_generator.pt')
        torch.save(conf_data['discriminator_model'].state_dict(),conf_data['save_model_path']+'/Seq/'+'pre_discriminator.pt') 
        
        conf_data['rollout'] = Rollout(generator, 0.8)

    for epoch in range(epochs):
        conf_data['epoch'] = epoch
        if seq == 0:
            to_iter = dataloader
        elif seq == 1: #TODO: Change this back to 1
            to_iter = [1]

        for i, iterator in enumerate(to_iter):
            optimizer_D = conf_data['discriminator_optimizer']
            optimizer_G = conf_data['generator_optimizer']

            generator = conf_data['generator_model']
            discriminator = conf_data['discriminator_model']

            g_loss_func = conf_data['generator_loss']
            d_loss_func = conf_data['discriminator_loss']

            # if aux = 1:
                

            #print ("Reached here --------------> ")
            conf_data['iterator'] = i 
            if seq == 0:
                
                if data_label == 1:
                    imgs,labels = iterator
                else:
                    imgs = iterator
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                conf_data['valid']=valid
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
                conf_data['fake']=fake
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                if data_label == 1:
                    labels = Variable(labels.type(LongTensor))
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], g_latent_dim))))
                if classes > 0:
                    gen_labels = Variable(LongTensor(np.random.randint(0, classes, imgs.shape[0])))
                    conf_data['gen_labels'] = gen_labels
            # elif seq == 1: #If yes seqGAN 
            #     # samples = generator.sample(mini_batch_size,conf_data['generator']['sequece_length'])
            #     # zeros = torch.zeros((mini_batch_size,1)).type(LongTensor)
            #     # imgs = Variable(torch.cat([zeros,samples.data]),dim=1)[:,:-1].contiguous() #TODO: change imgs to inps all, to make more sense of the code
            #     # targets = Variable(sample.data).contiguous().view((-1,))
            #     # rewards = rollout.get_reward(sample,16,discriminator)
            #     # rewards = Variable(Tensor(rewards))
            #     # prob = generator.forward(inputs)
            #     # loss = gen_gan_loss(prob)
            #     pass
            #     #optimizer_G
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            if seq == 1:#TODO change this back to 1
                dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, mini_batch_size)
            for i in range(temp): # TODO: Make this a parameter -> for x updates --> I am read the stored models here as well. Should I reamove this ???
                optimizer_D = conf_data['discriminator_optimizer']
                optimizer_G = conf_data['generator_optimizer']

                generator = conf_data['generator_model']
                discriminator = conf_data['discriminator_model']

                g_loss_func = conf_data['generator_loss']
                d_loss_func = conf_data['discriminator_loss']
                if classes <= 0:
                    #print ("Reached here 2 --------------> ")
                    if seq == 0:
                        gen_imgs = generator(z)
                        # Measure discriminator's ability to classify real from generated samples
                        #Real images
                        real_validity = discriminator(real_imgs)
                        #Fake images
                        fake_validity = discriminator(gen_imgs.detach())
                    if seq == 1:
                        generate_samples(generator,mini_batch_size,GENERATED_NUM,NEGATIVE_FILE,conf_data)
                        dis_data_iter = DisDataIter(POSITIVE_FILE,NEGATIVE_FILE,mini_batch_size)
                        loss = train_epoch(discriminator,dis_data_iter,d_loss_func,optimizer_D,conf_data,'d')
                        conf_data['d_loss'] = loss
                        #exit()

                else:
                    if seq == 0:
                        gen_imgs = generator(z,gen_labels)
                        real_validity = discriminator(real_imgs, labels)
                        fake_validity = discriminator(gen_imgs.detach(), labels)

                if seq == 0:
                    conf_data['gen_imgs'] = gen_imgs
                if seq == 0:
                    if w_loss == 0:
                        real_loss = d_loss_func.loss(real_validity, valid)
                        fake_loss = d_loss_func.loss(fake_validity, fake)
                        d_loss = (real_loss + fake_loss) / 2
                    elif w_loss == 1:
                        d_loss = -d_loss_func.loss(real_validity,valid) + d_loss_func.loss(fake_validity,fake)
                        if lambda_gp > 0:
                            conf_data['real_data_sample'] = real_imgs.data
                            conf_data['fake_data_sample'] = gen_imgs.data
                            conf_data = compute_gradient_penalty(conf_data)
                            gradient_penalty = conf_data['gradient_penalty']
                            d_loss = d_loss+ lambda_gp * gradient_penalty
                    conf_data['d_loss'] = d_loss
                    d_loss.backward()
                    optimizer_D.step()

                if clip_value > 0:
                    # Clip weights of discriminator
                    for p in discriminator.parameters():
                        p.data.clamp_(-clip_value, clip_value)

            # -----------------   
            #  Train Generator
            # -----------------
            conf_data['generator_model'] = generator
            conf_data['discriminator_model'] = discriminator

            #Next 4 lines were recently added maybe have to remove this.
            conf_data['optimizer_G'] = optimizer_G
            conf_data['optimizer_D'] = optimizer_D
            conf_data['generator_loss'] = g_loss_func
            conf_data['discriminator_loss'] = d_loss_func
            if seq == 0:
                conf_data['noise'] = z

            if n_critic <= 0:
                conf_data = training_fucntion_generator(conf_data)
            elif n_critic > 0:
                # Train the generator every n_critic iterations
                if i % n_critic == 0:
                    conf_data = training_fucntion_generator(conf_data)
            #exit()

           # print ("------------------ Here (train_GAN.py)")

            if seq == 0:
                batches_done = epoch * len(dataloader) + i
                if batches_done % int(conf_data['sample_interval']) == 0:
                    if classes <= 0:
                        # print ("Here")
                        # print (type(gen_imgs.data[:25]))
                        # print (gen_imgs.data[:25].shape)
                        save_image(gen_imgs.data[:25], conf_data['result_path']+'/%d.png' % batches_done, nrow=5, normalize=True) 
                    elif classes > 0:
                        sample_image(10,batches_done,conf_data)
        if seq == 0:
            log_file.write("[Epoch %d/%d] [D loss: %f] [G loss: %f] \n" % (epoch, epochs,conf_data['d_loss'].item(), conf_data['g_loss'].item()))           
        elif seq == 1:
            # print ("Done")
            log_file.write("[Epoch %d/%d] [D loss: %f] [G loss: %f] \n" % (epoch, epochs,conf_data['d_loss'], conf_data['g_loss']))           
    conf_data['generator_model'] = generator
    conf_data['discriminator_model'] = discriminator
    conf_data['log_file'] = log_file
    return conf_data