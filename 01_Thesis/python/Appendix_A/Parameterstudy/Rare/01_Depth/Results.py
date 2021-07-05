'''
Parameterstudy, Depth, Rare
'''


from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Fully_Connected/Depth/rare_depth.tex"
loc_chpt= "rom-using-autoencoders/01_Thesis/python/Appendix_A/Parameterstudy/Rare/01_Depth"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
####import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor

device = 'cpu'




class data():
    #load data
    f = np.load(join(home,loc_data))
    f = tensor(f, dtype=torch.float).to(device)

train_losses = []
val_losses = []
l2_losses = []
depth = [10,8,6,4,2]
min_idx = []

fig, ax  = plt.subplots(5)
fig.suptitle("Depth FCNN Rare")

for g in range(5):
    #g=4
    class params():
        N_EPOCHS = 2000
        BATCH_SIZE = 16
        INPUT_DIM = 40
        H_SIZES = [[40,20,10,5],[40,20,10],[40,20],[40],[]]
        H_SIZE = H_SIZES[g]
        h_layers = len(H_SIZES[g])
        LATENT_DIM = 3
        lr = 1e-4
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            sizes = [y for x in [[params.INPUT_DIM], params.H_SIZE] for y in x]
            for l in range(params.h_layers):
                self.add_module('layer_' + str(l), torch.nn.Linear(in_features=sizes[l],out_features=sizes[l+1]))
                # if sizes[l] != 40:
                self.add_module('activ_' + str(l), nn.LeakyReLU())
            self.add_module('layer_c',nn.Linear(in_features=sizes[-1], out_features=params.LATENT_DIM))
            self.add_module('activ_c', nn.Tanh())
        def forward(self, x):
            for _, method in self.named_children():
                x = method(x)
            return x



    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            sizes = [y for x in [[params.INPUT_DIM], params.H_SIZE] for y in x]
            sizes.reverse()
            self.add_module('layer_c',nn.Linear(in_features=params.LATENT_DIM, out_features=sizes[0]))
            self.add_module('activ_c', nn.LeakyReLU())
            for l in range(params.h_layers):
                self.add_module('layer_' + str(l), nn.Linear(in_features=sizes[l],out_features=sizes[l+1]))
                if sizes[l] != 40:
                    self.add_module('activ_' + str(l), nn.LeakyReLU())
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

    checkpoint = torch.load(join(home,loc_chpt,
        'Results/LS_{}.pt'.format(g)),
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



    ax[g].semilogy(train_loss,'k''--',label='Train')
    ax[g].semilogy(val_loss,'k''-',label='Test')
    ax[g].set_xlabel('Epoch')
    ax[g].set_ylabel('MSE Loss')
    ax[g].set_ylim(ymax=1e-5)
    i = [10,8,6,4,2]
    ax[g].set_title('{} Layer'.format(i[g]))
    ax[g].legend()
####tikzplotlib.save(join(home,loc_plot))
plt.show()

loss_dict = {"Depth":depth,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict,dtype=float)
print("Experiment FCNN Depth Rare")
print(loss_dict)

