import math
from torch.autograd import Variable
def train_epoch(model, data_iter, criterion, optimizer,conf_data,indicator):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        # print ("Data Here")
        # print (data)
        # print ("Target Here")
        # print (data)
        # exit()
        data = Variable(data)
        target = Variable(target)

        if conf_data['cuda']:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion.loss(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    if indicator == 'd':
        conf_data['discriminator_model'] = model
        conf_data['discriminator_loss'] = criterion
        conf_data['discriminator_optimizer'] = optimizer
    elif indicator == 'g':
        conf_data['generator_model'] = model
        conf_data['generator_loss'] = criterion
        conf_data['generator_optimizer'] = optimizer
    return math.exp(total_loss / total_words)