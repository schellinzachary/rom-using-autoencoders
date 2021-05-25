'''
Final Model, hydro
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p00001_4D_unshuffled.npy"
#loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Fully_Connected/Activations/hydro_act.tex"


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'



class data():
    #load data
    f = np.load(join(home,loc_data))
    f = tensor(f, dtype=torch.float).to(device)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.m = nn.ZeroPad2d((0,0,1,1))
        self.convE1 = nn.Conv2d(1,32,(6,10),stride=(3,10))
        self.convE2 = nn.Conv2d(32,64,(4,10),stride=(4,10))
        self.linearE1 = nn.Linear(in_features=256,out_features=3)
        self.add_module('act',nn.SiLU())

    def forward(self, x):
        x = self.m(x)
        x = self.act(self.convE1(x))
        x = self.act(self.convE2(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.linearE1(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linearD1 = nn.Linear(in_features=3, out_features=256)
        self.convD1 = nn.ConvTranspose2d(64,32,(4,10),stride=(4,10))
        self.convD2 = nn.ConvTranspose2d(32,1,(4,10),stride=(3,10))
        self.add_module('act',nn.SiLU())


    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,64,2,2])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z = self.enc(x)
        x = self.dec(z)
        return x, z

#encoder
encoder = Encoder()

#decoder
decoder = Decoder()

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

checkpoint = torch.load('Results/1.pt')

model.load_state_dict(checkpoint['model_state_dict'])
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']
N_EPOCHS = checkpoint['epoch']



v = sio.loadmat('/home/zachi/ROM_using_Autoencoders/02_data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/zachi/ROM_using_Autoencoders/02_data_sod/sod25Kn0p00001/t.mat')
x = sio.loadmat('/home/zachi/ROM_using_Autoencoders/02_data_sod/sod25Kn0p00001/x.mat')
x = x['x']
x = x.squeeze()
t  = t['treport']
v = v['v']
t=t.squeeze()
t=t.T



rec, z = model(data.f)

l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
print(l2_error)

def plot_code(z,v):
    fig , ax = plt.subplots(1,3)
    for i in range(3):
        ax[i].plot(v,z[:,i].detach().numpy(),'k',label='c_{}'.format(i))
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('c_{}'.format(i))
        ax[i].legend()
    #tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/Hydro/ConvCode.tex')
    plt.show()

def plot_results(rec,c):
    fig,ax = plt.subplots(3,1)
    a = ax[0].imshow(rec[39,0,:,:].detach().numpy())
    b = ax[1].imshow(data.f[39,0,:,:].detach().numpy())

def plot_training(train_loss,test_loss):
    plt.figure()
    plt.semilogy(train_losses,'k''--',label='Train')
    plt.semilogy(test_losses,'k''-',label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    ax = plt.gca()
    plt.legend()
    # tikzplotlib.save('/home/fusilly/ROM_using_Autoencoders/Bachelorarbeit/Figures/Layer_Sizes/{}.tex'.format(g))
    plt.show()


#plot_code(z,v)
plot_training(train_losses,test_losses)

