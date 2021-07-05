'''
Activations, Rare
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Fully_Connected/Activations/hydro_act.tex"
loc_chpt = "rom-using-autoencoders/01_Thesis/python/Appendix_A/Parameterstudy/Hydro/04_Activations"


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
###import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'
print(device)

#set variables
activations = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'silu': nn.SiLU(),
    'tanh': nn.Tanh(),
    'leaky': nn.LeakyReLU() 
}


#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)

class Encoder(nn.Module):
    def __init__(self, a, c):
        super(Encoder, self).__init__()
        self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=30))
        self.add_module('activ_1', a)
        self.add_module('layer_c',nn.Linear(in_features=30, out_features=3))
        self.add_module('activ_c', c)
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder(nn.Module):
    def __init__(self,a):
        super(Decoder, self).__init__()
        self.add_module('layer_c',nn.Linear(in_features=3, out_features=30))
        self.add_module('activ_c', a)
        self.add_module('layer_4', nn.Linear(in_features=30,out_features=40))
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


experiments = (
    ('relu','relu'),
    ('elu','elu'),
    ('tanh','tanh'),
    ('silu','silu'),
    ('leaky','leaky'),
    ('elu','tanh'),
    ('leaky','tanh'),
    ('elu','silu')
    )

best_models = (
    "('relu', 'relu')-epoch4989-val_loss9.800E-09",
    "('elu', 'elu')-epoch4990-val_loss4.454E-09",
    "('tanh', 'tanh')-epoch4990-val_loss7.835E-09",
    "('silu', 'silu')-epoch4990-val_loss7.693E-09",
    "('leaky', 'leaky')-epoch4990-val_loss1.867E-08",
    "('elu', 'tanh')-epoch4990-val_loss5.503E-09",
    "('leaky', 'tanh')-epoch4990-val_loss1.004E-08",
    "('elu', 'silu')-epoch4990-val_loss8.115E-09"   
    )




train_losses = []
val_losses = []
l2_losses = []
act = []
min_idx = []

fig, ax = plt.subplots(8,1)
fig.suptitle("Activations FCNN Hy")


for idx, (ac_combo, best_model) in enumerate(zip(experiments,best_models)):
    a, c = ac_combo
    a = activations[a]
    c = activations[c]
    #encoder
    encoder = Encoder(a,c)

    #decoder
    decoder = Decoder(a)

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    checkpoint_model = torch.load(join(home,loc_chpt,
        'Results/{}.pt'.format(best_model)),
    map_location="cpu")
    checkpoint_loss = torch.load(join(home,loc_chpt,
        'Results/last-{}.pt'.format(ac_combo)),
    map_location="cpu")
    model.load_state_dict(checkpoint_model['model_state_dict'][0])
    train_loss = checkpoint_loss['train_losses']
    val_loss = checkpoint_loss['test_losses']


    rec = model(f)
    l2_loss = torch.norm((f - rec).flatten())/torch.norm(f.flatten())

    train_losses.append(np.min(train_loss))
    val_losses.append(np.min(val_loss))
    l2_losses.append(l2_loss.detach().numpy())
    min_idx.append(val_loss.index(min(val_loss)))
    act.append(ac_combo)
    
    ax[idx].semilogy(train_loss,'k''--',label='Train')
    ax[idx].semilogy(val_loss,'k''-',label='Test')
    ax[idx].set_xlabel('Epoch')
    ax[idx].set_ylabel('MSE Loss')
    ax[idx].set_title('{} '.format(ac_combo))
    ax[idx].set_ylim(ymax=1e-5)
    ax[idx].legend()

###tikzplotlib.save(join(home,loc_plot))


loss_dict = {"act":act,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict)
print("Experiment FCNN Activations Hydro")
print(loss_dict)
plt.show()