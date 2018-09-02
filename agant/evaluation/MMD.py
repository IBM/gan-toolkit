import torch

class Metric:
    def __init__(self):
        super(Metric, self).__init__()
        self.alpha = 0.05

    def calculate(self,true_data,generated_data):
        """
        MMD for given real dataset and generated dataset.
        
        Parameters
        ----------
        true_data: array
            Torch array containing the real dataseta.

        generated_data: array
            Torch array containing the generated dataseta.       

        Returns
        -------
        output.data.numpy(): int
            MMD evaluation score.   

        """
        B = true_data.size(0)
        x = true_data.view(true_data.size(0),true_data.size(2)*true_data.size(3))
        y = generated_data.view(generated_data.size(0), generated_data.size(2) * generated_data.size(3))
        #print (x.shape)
        
        '''
        TODO:
        -> Ask about alpha.
        '''

        x = x[:250]
        y = y[:250]

        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        temp = xx.diag().unsqueeze(0).expand_as(xx)
        #print (temp.shape)

        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K = torch.exp(- self.alpha * (rx.t() + rx - 2*xx))
        L = torch.exp(- self.alpha * (ry.t() + ry - 2*yy))
        P = torch.exp(- self.alpha * (rx.t() + ry - 2*zz))

        beta = (1./(B*(B-1)))
        gamma = (2./(B*B)) 

        output = beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)
        #print (output.data.numpy())

        return output.data.numpy()