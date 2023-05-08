import torch
from torch import nn
import numpy as np



#For VAE
def kld_loss(mu, log_var): 
    var = log_var.exp()
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - var, dim=1), dim=0)
    return kl_loss


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def vae_split(latent):
    size = latent.shape[1] // 2 # divide the latent representation into mu and log_var
    mu = latent[:, :size] 
    log_var = latent[:, size:]  
    return mu, log_var


def vae_loss_handler(data, recons, latent, kld_weight=8e-5, *args, **kwargs):
    mu, log_var = vae_split(latent)
    kl_loss = kld_loss(mu, log_var)
    return kld_weight * kl_loss + LogCoshLoss()(recons, data) #F.mse_loss(recons, data)
#################################





# For Real-Bogus classifier
class TverskyLoss(nn.Module):
    def __init__(self, alfa=0.3, smooth=1, focal_tl=False, gamma=0.75):
        """
        """
        super(TverskyLoss, self).__init__()

        self.smooth = smooth
        self.alpha = alfa
        self.beta = 1 - alfa
        self.focal = focal_tl
        self.gamma = gamma

    def forward(self, nn_output, gt):
        gt_onehot = torch.zeros(nn_output.shape[0], 2).to(nn_output.device.type)
        for i, lab in enumerate(gt):
            gt_onehot[i][lab] = 1
         
        tp = torch.sum(gt_onehot * nn_output)
        fn = torch.sum((1 - nn_output) * gt_onehot)
        fp = torch.sum((1 - gt_onehot) * nn_output)
        TI = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        if self.focal:
            return torch.pow((1 - TI), self.gamma)
        else:
            return 1 - TI


def rnn_loss_handler(nn_out, gt):
    #prob = nn.Sigmoid()(nn_out)
    return 0.7 * nn.CrossEntropyLoss()(nn_out, gt) + 0.3 * TverskyLoss()(nn_out, gt)
