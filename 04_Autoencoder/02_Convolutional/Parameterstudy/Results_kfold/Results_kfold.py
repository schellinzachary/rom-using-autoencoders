
'''
Parameterstudy_01_Layer_Size
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/flow_4D.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Convolutional/kfold.tex"

import pandas as pd
import numpy as np
#import tikzplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as tensor

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'

#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convE1 = nn.Conv2d(1,8,(5,5),stride=(5,5))
        self.convE2 = nn.Conv2d(8,16,(5,5),stride=(5,5))
        self.linearE1 = nn.Linear(in_features=128,out_features=5)
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.act(self.convE1(x))
        x = self.act(self.convE2(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.linearE1(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linearD1 = nn.Linear(in_features=5, out_features=128)
        self.convD1 = nn.ConvTranspose2d(16,8,(5,5),stride=(5,5))
        self.convD2 = nn.ConvTranspose2d(8,1,(5,5),stride=(5,5))
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,8])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc= enc
        self.dec = dec

    def forward(self, x):
        z = self.enc(x)
        x = self.dec(z)
        return x


best_models = (
    "fold0-epoch990-val_loss4.527E-05",
    "fold1-epoch989-val_loss1.268E-05",
    "fold2-epoch989-val_loss4.958E-05",
    "fold3-epoch990-val_loss1.491E-05",
    "fold4-epoch988-val_loss4.234E-05", 
    )

train_losses = []
val_losses = []
l2_losses = []
variable = []
min_idx = []

fig, ax = plt.subplots(1,5)

for idx, best_model in enumerate(best_models):

    #encoder
    encoder = Encoder()
    #decoder
    decoder = Decoder()
    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    checkpoint_model = torch.load('Results/{}.pt'.format(best_model))
    checkpoint_loss = torch.load('Results/last-fold-{}.pt'.format(idx))
    model.load_state_dict(checkpoint_model['model_state_dict'])
    train_loss = checkpoint_loss['train_losses']
    val_loss = checkpoint_loss['test_losses']

    rec = model(f)

    l2_loss = torch.norm((f - rec).flatten())/torch.norm(f.flatten())

    train_losses.append(np.min(train_loss))
    val_losses.append(np.min(val_loss))
    l2_losses.append(l2_loss.detach().numpy())
    min_idx.append(val_loss.index(min(val_loss)))
    variable.append(idx)
    
    mean = np.mean(val_losses)
    std = np.std(val_losses)
    var = np.var(val_losses)


    ax[idx].semilogy(train_loss,'k''--',label='Train')
    ax[idx].semilogy(val_loss,'k''-',label='Test')
    ax[idx].set_xlabel('Epoch')
    ax[idx].set_ylabel('MSE Loss')
    ax[idx].set_title('fold{} '.format(idx))
    ax[idx].set_ylim(ymax=1e-2)
    ax[idx].legend()

#tikzplotlib.save(join(home,loc_plot))


loss_dict = {
    "fold":variable,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict)

print(loss_dict)
print()
print("Mean of val loss:",mean)
print()
print("Variance of val loss:",var)
print()
print("Standard deviation of val loss:", std)
plt.show()