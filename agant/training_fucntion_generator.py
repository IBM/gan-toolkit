import torch
import torch.nn as nn
from torch.autograd import Variable
from sequential_generate_sample import generate_samples
from data_iter import GenDataIter, DisDataIter
from sequential_eval_epoch import eval_epoch
from target_lstm import TargetLSTM


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable 
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss
def training_fucntion_generator(conf_data):
    """Training Process for generator network.
    
    Parameters
    ----------
    conf_data: dict
        Dictionary containing all parameters and objects.       

    Returns
    -------
    conf_data: dict
        Dictionary containing all parameters and objects.       

    """
    PRE_EPOCH_NUM = 2
    seq = conf_data['GAN_model']['seq']
    BATCH_SIZE = 64
    GENERATED_NUM = 10000 
    EVAL_FILE = 'eval.data'
    POSITIVE_FILE = 'real.data'
    NEGATIVE_FILE = 'gene.data'
    
    classes = int(conf_data['GAN_model']['classes'])
    w_loss = int(conf_data['GAN_model']['w_loss'])
    g_loss_func = conf_data['generator_loss']
    
    epoch = conf_data['epoch']
    epochs = conf_data['epochs']

    generator = conf_data['generator_model']
    discriminator = conf_data['discriminator_model']
    optimizer_G = conf_data['generator_optimizer']
    mini_batch_size = (conf_data['GAN_model']['mini_batch_size'])

    optimizer_G.zero_grad()

    # Generate a batch of images
    if seq == 0:
        valid = conf_data['valid']
        gen_imgs = conf_data['gen_imgs']
        z = conf_data['noise']
        if classes <= 0:
            #gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs)
        elif classes > 0:
            gen_labels = conf_data['gen_labels']
            #gen_imgs = generator(z,gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
           
        if w_loss == 1:
            g_loss = -g_loss_func.loss(validity,valid)
        elif w_loss == 0:
            g_loss = g_loss_func.loss(validity,valid) 
        conf_data['g_loss'] = g_loss
        g_loss.backward()
        optimizer_G.step()
    elif seq == 1:
        #print ("Reached Here 3 ---------> ")
        gen_gan_loss = GANLoss()
        rollout = conf_data['rollout']
        target_lstm = conf_data['target_lstm']
        for it in range(1):
            samples = generator.sample(mini_batch_size, conf_data['generator']['sequece_length'])
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((mini_batch_size, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            if conf_data['cuda']:
                rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
            prob = generator.forward(inputs)
            rewards = rewards.contiguous().view(-1,)
            loss = gen_gan_loss(prob, targets, rewards)
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
        #TODO : Change back. Uncomment and indent till line above to rollout
        #if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
        generate_samples(generator, mini_batch_size, GENERATED_NUM, EVAL_FILE,conf_data)
        #print ("Reached Here 4 ---------> ")
        eval_iter = GenDataIter(EVAL_FILE, mini_batch_size)
        #print ("Reached Here 5 ---------> ")
        loss = eval_epoch(target_lstm, eval_iter, g_loss_func,conf_data)
        conf_data['g_loss']= loss
        #print ("Reached Here 6 ---------> ")
       #print('Batch [%d] True Loss: %f' % (total_batch, loss))
        rollout.update_params()

    #g_loss = g_loss_func.loss(validity, valid)

    # print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, epochs, conf_data['iterator'], 5,
    #                                                 conf_data['d_loss'].item(), g_loss.item()))
    if seq == 0:
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, epochs, conf_data['iterator'], len(conf_data['data_learn']),
                                                       conf_data['d_loss'].item(), g_loss.item()))
    elif seq == 1:
        print("[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f]"% (epoch, epochs, conf_data['iterator'],
                                                       conf_data['d_loss'], conf_data['g_loss']))
    #print ("Done")

    conf_data['generator_model'] = generator
    conf_data['generator_optimizer'] = optimizer_G

    conf_data['discriminator_model'] = discriminator
    conf_data['generator_loss'] = g_loss_func
    if seq == 1:
        conf_data['rollout'] = rollout
    return conf_data