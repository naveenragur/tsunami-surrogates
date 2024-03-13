r"""
Train variational autoencoder

"""
import glob, os
import shutil
import numpy as np
import vae as vae
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
# from torch.autograd import Variable

# Environment variable
try:
    home = os.environ['nragu']
    zdim2use = int(os.environ['zdim2use'])
    vae2use = int(os.environ['vae2use'])
    kfold2use = int(os.environ['kfold2use'])
    nepochs2use = int(os.environ['nepochs2use'])
    print('\n nepochs:', nepochs2use, 'zdim2use:', zdim2use,'vae2use:', vae2use,'kfold2use:', kfold2use)
except:
    raise Exception('*** Must first set environment variable***')

# def loss_function(recon_x, x, mu, logvar) -> Variable:
def loss_function(recon_x, x, mu, logvar):
    # how well do input x and output recon_x agree?
    BCE = F.mse_loss(recon_x, x) 

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= (recon_x.shape[0] * recon_x.shape[1]) #was just sum of dimensions/number of entries
    #KLD /= recon_x.shape[0] * recon_x.shape[-1]
    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    # print(BCE.detach().numpy(), KLD.detach().numpy())
    return 3*BCE + KLD

if __name__ == "__main__":

    # vae model definition
    VAE = vae.VarAutoEncoder(gauges=[5010,5015,5020,5675], model_name='vae_riku')

    # use GPU for training (e.g. on cluster)
    VAE.device = 'cuda'

    #run loop of folds and calculate the average accuracy and plot training losses

    # load data
    VAE.load_data(batch_size=1000, data_fname='_data/riku.npy',kfold=kfold2use)

    # # train ensemble
    VAE.train_vae(vaeruns=vae2use,
                    zdims=zdim2use,
                    torch_loss_func=loss_function,
                    torch_optimizer=optim.Adam,
                    nepochs=nepochs2use,
                    input_gauges=[5010,5015,5020],
                    kfold=kfold2use)

  # identify the epochs where the loss is minimum
    train_loss = np.load('_output/vae_riku_train_loss_k{:d}.npy'.format(kfold2use))
    test_loss = np.load('_output/vae_riku_test_loss_k{:d}.npy'.format(kfold2use))
    
    # training loss plot
    s0 = 2
    fig, ax = plt.subplots(figsize=(s0 * 6, s0 * 4))
    line0, = ax.semilogy(train_loss[1:],color='blue',linewidth=1, label='train loss')
    line0, = ax.semilogy(test_loss[1:],color='green',linewidth=1,label='test loss')
    ax.grid(which='minor', alpha=0.05)
    ax.grid(which='major', alpha=0.1)
    ax.xaxis.set_ticks(np.arange(0, nepochs2use, 500))
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    ax.set_xlim([0, nepochs2use])
    
    fig.tight_layout()

    fname = str(1024) + '_vae_training_losses_k{:d}.png'.format(kfold2use)
    save_fname = os.path.join('_output', fname)
    fig.savefig(save_fname)
