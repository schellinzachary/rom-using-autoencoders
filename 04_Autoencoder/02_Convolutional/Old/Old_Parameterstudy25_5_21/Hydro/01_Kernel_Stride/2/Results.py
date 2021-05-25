'''
Kernel, Hydro
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

device='cpu'

class data():
    #load data
    f = np.load(join(home,loc_data))
    f = tensor(f, dtype=torch.float).to(device)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.m = nn.ZeroPad2d((0,0,1,1))
        self.convE1 = nn.Conv2d(1,8,(6,10),stride=(3,10))
        self.convE2 = nn.Conv2d(8,16,(4,10),stride=(4,10))
        self.linearE1 = nn.Linear(in_features=64,out_features=3)
        self.act = nn.Tanh()
        #self.act_c = nn.Tanh()

    def forward(self, x):
        x = self.m(x)
        x = self.act(self.convE1(x))
        x = self.act(self.convE2(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        #x = self.act_c(self.linearE1(x))
        x = self.linearE1(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linearD1 = nn.Linear(in_features=3, out_features=64)
        self.convD1 = nn.ConvTranspose2d(16,8,(4,10),stride=(4,10))
        self.convD2 = nn.ConvTranspose2d(8,1,(4,10),stride=(3,10))
        self.act = nn.Tanh()
        #self.act_c = nn.Tanh()

    def forward(self, x):
        x = self.linearD1(x)
        #x = self.act_c(self.linearD1(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,2,2])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

rec = model(data.f)

l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
print(l2_error)

rec = model(data.f)
fig,ax = plt.subplots(3,1)
a = ax[0].imshow(rec[39,0,:,:].detach().numpy())
b = ax[1].imshow(data.f[39,0,:,:].detach().numpy())



# plt.figure(i)
# plt.semilogy(train_losses,'k''--',label='Train')
# plt.semilogy(test_losses,'k''-',label='Test')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# ax = plt.gca()
# # plt.title('{}'.format((act_list[i],act_c_list[i])))
# plt.title('{}'.format(act_list[i]))
# plt.legend()

plt.figure()
plt.semilogy(train_losses,'k''--',label='Train')
plt.semilogy(test_losses,'k''-',label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
ax = plt.gca()
plt.legend()
# tikzplotlib.save('/home/fusilly/ROM_using_Autoencoders/Bachelorarbeit/Figures/Layer_Sizes/{}.tex'.format(g))
plt.show()

