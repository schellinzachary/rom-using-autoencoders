'''
Parameterstudy, Width, Hydro
'''

loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Fully_Connected/Width/hydro_width.tex"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
#####import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor

from os.path import join
from pathlib import Path
home = Path.home()
loc_chpt = "rom-using-autoencoders/01_Thesis/python/Appendix_A/Parameterstudy/Hydro/02_Layer_Size"



device = 'cpu'


class params():
    BATCH_SIZE = 16
    INPUT_DIM = 40
    H_SIZES = [50,40,30,20,10]
    LATENT_DIM = 3

class data():
    #load data
    f = np.load(join(home,loc_data))
    f = tensor(f, dtype=torch.float)#.to(device)

class Encoder(nn.Module):
    def __init__(self,h_size):
        super(Encoder, self).__init__()
        self.h_size = h_size
        self.add_module('layer_1', torch.nn.Linear(params.INPUT_DIM,self.h_size))
        self.add_module('activ_1', nn.LeakyReLU())
        self.add_module('layer_c',nn.Linear(self.h_size, params.LATENT_DIM))
        self.add_module('activ_c', nn.Tanh())
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder(nn.Module):
    def __init__(self,h_size):
        super(Decoder, self).__init__()
        self.h_size = h_size
        self.add_module('layer_c',nn.Linear(params.LATENT_DIM, self.h_size))
        self.add_module('activ_c', nn.LeakyReLU())
        self.add_module('layer_4', nn.Linear(self.h_size, params.INPUT_DIM))
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z = self.enc(x)
        predicted = self.dec(z)
        return predicted


train_losses = []
val_losses = []
l2_losses = []
width = []
min_idx = []

fig, ax  = plt.subplots(5)
fig.suptitle("Layers FCNN Hy")

for idx, i in enumerate(params.H_SIZES):
    #encoder
    encoder = Encoder(i)

    #decoder
    decoder = Decoder(i)

    #Autoencoder
    model = Autoencoder(encoder, decoder)#.to(device)

    checkpoint = torch.load(join(home,loc_chpt,
        'Results/P%s.pt'%i),
    map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    train_loss = checkpoint['train_losses']
    val_loss = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']

    rec = model(data.f)
    l2_loss = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())

    train_losses.append(np.min(train_loss))
    val_losses.append(np.min(val_loss))
    l2_losses.append(l2_loss.detach().numpy())
    min_idx.append(val_loss.index(min(val_loss)))
    width.append(i)

  
    
    ax[idx].semilogy(train_loss,'k''--',label='Train')
    ax[idx].semilogy(val_loss,'k''-',label='Test')
    ax[idx].set_xlabel('Epoch')
    ax[idx].set_ylabel('MSE Loss')
    ax[idx].set_title('%s Nodes'%i)
    ax[idx].set_ylim(ymax=1e-5)
    ax[idx].legend()
####tikzplotlib.save(join(home,loc_plot))
plt.show()

loss_dict = {"Width":width,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict,dtype=float)
print("Experiment FCNN Layer Hydro")
print(loss_dict)