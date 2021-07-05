'''
Parameterstudy, Batch Size, Rare
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Fully_Connected/Batch_Size/rare_batch.tex"
loc_chpt= "rom-using-autoencoders/01_Thesis/python/Appendix_A/Parameterstudy/Rare/03_Batch_Size"


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
####import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor

device='cpu'


class params():
    INPUT_DIM = 40
    H_SIZES = 40
    LATENT_DIM = 5

class data():
    #load data
    f = np.load(join(home,loc_data))
    f = tensor(f, dtype=torch.float).to(device)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.add_module('layer_1', torch.nn.Linear(in_features=params.INPUT_DIM,out_features=params.H_SIZES))
        self.add_module('activ_1', nn.LeakyReLU())
        self.add_module('layer_c',nn.Linear(in_features=params.H_SIZES, out_features=params.LATENT_DIM))
        self.add_module('activ_c', nn.Tanh())
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.add_module('layer_c',nn.Linear(in_features=params.LATENT_DIM, out_features=params.H_SIZES))
        self.add_module('activ_c', nn.LeakyReLU())
        self.add_module('layer_4', nn.Linear(in_features=params.H_SIZES,out_features=params.INPUT_DIM))
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

#encoder
encoder = Encoder()

#decoder
decoder = Decoder()

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

model = Autoencoder(encoder, decoder).to(device)

fig, ax = plt.subplots(5,1)

train_losses = []
val_losses = []
l2_losses = []
batch_s = []
min_idx = []

for idx, batch in enumerate([32,16,8,4,2]):

    checkpoint = torch.load(join(home,loc_chpt,
        'Results/{}.pt'.format(batch))
    , map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    train_loss = checkpoint['train_losses']
    val_loss = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']
    batch_size = checkpoint['batch_size']

    rec = model(data.f)
    l2_loss = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())

    train_losses.append(np.min(train_loss))
    val_losses.append(np.min(val_loss))
    l2_losses.append(l2_loss.detach().numpy())
    min_idx.append(val_loss.index(min(val_loss)))
    batch_s.append(batch)
    
    ax[idx].semilogy(train_loss,'k''--',label='Train')
    ax[idx].semilogy(val_loss,'k''-',label='Test')
    ax[idx].set_xlabel('Epoch')
    ax[idx].set_ylabel('MSE Loss')
    ax[idx].set_title('Batch Size %s '%batch)
    ax[idx].set_ylim(ymax=1e-5)
    ax[idx].legend()

####tikzplotlib.save(join(home,loc_plot))



loss_dict = {"Batch Size":batch_s,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict,dtype=float)
print("Experiment FCNN Batch Rare")
print(loss_dict)
plt.show()
