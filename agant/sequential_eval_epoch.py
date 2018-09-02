from torch.autograd import Variable
import math
def eval_epoch(model, data_iter, criterion,conf_data):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if conf_data['cuda']:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion.loss(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
    data_iter.reset()
    return math.exp(total_loss / total_words)