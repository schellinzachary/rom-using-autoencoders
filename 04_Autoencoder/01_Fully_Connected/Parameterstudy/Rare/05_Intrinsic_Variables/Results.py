'''
Intrinsic Variables, Rare
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p01_2D.npy"
#loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Fully_Connected/Activations/hydro_act.tex"


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import tikzplotlib

import torch
import torch.nn as nn
import torch.tensor as tensor

device='cpu'


#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)

class Encoder(nn.Module):
    def __init__(self,int_var=None):
        super(Encoder, self).__init__()
        self.int_var = int_var
        self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=40))
        self.add_module('activ_1', nn.ReLU())
        self.add_module('layer_c',nn.Linear(in_features=40, out_features=self.int_var))
        self.add_module('activ_c', nn.ReLU())
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder(nn.Module):
    def __init__(self,int_var=None):
        super(Decoder, self).__init__()
        self.int_var = int_var
        self.add_module('layer_c',nn.Linear(in_features=self.int_var, out_features=40))
        self.add_module('activ_c', nn.ReLU())
        self.add_module('layer_4', nn.Linear(in_features=40,out_features=40))
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



models = {
    1 : "int_var-1-epoch1387-val_loss2.350E-03.pt",
    2 : "int_var-2-epoch4990-val_loss2.051E-07.pt",
    4 : "int_var-4-epoch4988-val_loss7.083E-08.pt",
    8 : "int_var-8-epoch4980-val_loss1.168E-08.pt",
    16: "int_var-16-epoch4990-val_loss9.933E-10.pt",
    32: "int_var-32-epoch4990-val_loss1.116E-09.pt"
    }



train_losses = []
val_losses = []
l2_losses = []
act = []
min_idx = []

fig, ax = plt.subplots(6,1)
fig2, ax2 = plt.subplots(6,1)

for idx, iv in enumerate([1,2,4,8,16,32]):
    
    #encoder
    encoder = Encoder(iv)

    #decoder
    decoder = Decoder(iv)

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    checkpoint_model = torch.load('Results/%s'%models[iv],map_location="cpu")
    checkpoint_loss = torch.load('Results/last-%s.pt'%iv,
        map_location="cpu")
    model.load_state_dict(checkpoint_model['model_state_dict'])
    #model.load_state_dict(checkpoint_loss['model_state_dict'])
    train_loss = checkpoint_loss['train_losses']
    val_loss = checkpoint_loss['test_losses']


    rec = model(f)
    print(torch.sum(rec))
    l2_loss = torch.norm((f - rec).flatten())/torch.norm(f.flatten())

    train_losses.append(np.min(train_loss))
    val_losses.append(np.min(val_loss))
    l2_losses.append(l2_loss.detach().numpy())
    min_idx.append(val_loss.index(min(val_loss)))
    act.append(iv)
    
    ax[idx].semilogy(train_loss,'k''--',label='Train')
    ax[idx].semilogy(val_loss,'k''-',label='Test')
    ax[idx].set_xlabel('Epoch')
    ax[idx].set_ylabel('MSE Loss')
    ax[idx].set_title('%s '%iv)
    ax[idx].set_ylim(ymax=1e-5)
    ax[idx].legend()

    #ax2[idx].imshow(rec[0,:,:].squeeze().detach().numpy())
    #ax2[idx].set_titĺe('{}'.format(ac_combo))

#tikzplotlib.save(join(home,loc_plot))


loss_dict = {"iv":act,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict)

print(loss_dict)
plt.show()