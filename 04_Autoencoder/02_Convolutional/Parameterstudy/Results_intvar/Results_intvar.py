'''
Intrinsic Variales Variation
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/flow_4D.npy"
#loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Convolutional/4l_activations.tex"


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
#import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor


torch.manual_seed(42)
device = 'cpu'

class Encoder_2(nn.Module):
    def __init__(self, iv):
        super(Encoder_2, self).__init__()
        self.iv = iv
        self.convE1 = nn.Conv2d(
            1,8,(5,5),
            stride=(5,5)
            )
        self.convE2 = nn.Conv2d(
            8,16,(5,5),
            stride=(5,5)
            )
        self.linearE1 = nn.Linear(in_features=128,out_features=self.iv)
        self.add_module('act_a', nn.ELU())
        self.add_module('act_c', nn.SiLU())


    def forward(self, x):
        x = self.act_a(self.convE1(x))
        x = self.act_a(self.convE2(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.act_a(self.linearE1(x))
        return x

class Decoder_2(nn.Module):
    def __init__(self, iv):
        super(Decoder_2, self).__init__()
        self.iv = iv
        self.linearD1 = nn.Linear(
            in_features=self.iv,
            out_features=128
            )
        self.convD1 = nn.ConvTranspose2d(
            16,8,(5,5),
            stride=(5,5)
            )
        self.convD2 = nn.ConvTranspose2d(
            8,1,(5,5),
            stride=(5,5)
            )
        self.add_module('act_a', nn.ELU())


    def forward(self, x):
        x = self.act_a(self.linearD1(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,8])
        x = self.act_a(self.convD1(x))
        x = self.convD2(x)
        return x


class Encoder_4(nn.Module):
    def __init__(self,a=None ,c=None):
        super(Encoder_4, self).__init__()
        self.convE1 = nn.Conv2d(
            1,8,(3,3),
            stride=(3,3),
            padding=(3,2)
            )
        self.convE2 = nn.Conv2d(
            8,16,(3,3),
            stride=(3,3),
            padding=(3,2),
            )
        self.convE3 = nn.Conv2d(
            16,32,(3,3),
            stride=(3,3),
            padding=(3,2)
            )
        self.convE4 = nn.Conv2d(
            32,64,(3,3),
            stride=(3,3),
            padding=(0,0)
            )
        self.linearE1 = nn.Linear(in_features=192,
            out_features=5
            )
        self.add_module('act_a', a)
        self.add_module('act_c', c)


    def forward(self, x):
        x = self.act_a(self.convE1(x))
        x = self.act_a(self.convE2(x))
        x = self.act_a(self.convE3(x))
        x = self.act_a(self.convE4(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.act_c(self.linearE1(x))
        return x

class Decoder_4(nn.Module):
    def __init__(self, a=None):
        super(Decoder_4, self).__init__()
        self.linearD1 = nn.Linear(in_features=5,
        out_features=192
        )
        self.convD1 = nn.ConvTranspose2d(
            64,32,(3,3),
            stride=(3,3),
            padding=(0,0),
            output_padding=(0,0)
            )
        self.convD2 = nn.ConvTranspose2d(
            32,16,(3,3),
            stride=(3,3),
            padding=(2,2),
            output_padding=(0,1)
            )
        self.convD3 = nn.ConvTranspose2d(
            16,8,(3,3),
            stride=(3,3),
            padding=(3,2),
            output_padding=(0,0)
            )
        self.convD4 = nn.ConvTranspose2d(
            8,1,(3,3),
            stride=(3,3),
            padding=(1,2)
            )
        self.add_module('act_a', a)


    def forward(self, x):
        x = self.act_a(self.linearD1(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,64,1,3])
        x = self.act_a(self.convD1(x))
        x = self.act_a(self.convD2(x))
        x = self.act_a(self.convD3(x))
        x = self.convD4(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x  




#uncomment for models for 2 Layer Autoencoder
models = {
    1 : "model0-int1-epoch1554-val_loss2.010E-04.pt",
    2 : "model0-int2-epoch1990-val_loss7.078E-05.pt",
    3 : "model0-act-('elu', 'silu')-epoch1974-val_loss5.696E-06.pt",
    4 : "model0-int4-epoch1734-val_loss6.353E-06.pt",
    8 : "model0-int8-epoch1752-val_loss6.601E-06.pt",
    16: "model0-int16-epoch1661-val_loss6.285E-06.pt",
    32: "model0-int32-epoch1741-val_loss6.148E-06.pt"
}


#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)


train_losses = []
val_losses = []
l2_losses = []
act = []
min_idx = []

fig, ax = plt.subplots(6,1)
fig2, ax2 = plt.subplots(6,1)

for idx, iv in enumerate([1,2,4,8,16,32]):
    
    #encoder
    encoder = Encoder_2(iv) #change to Encoder_4(a,c)

    #decoder
    decoder = Decoder_2(iv) #change to Decoder_4(a)

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    checkpoint_model = torch.load('noaugment/%s'%models[iv],map_location="cpu")
    checkpoint_loss = torch.load('noaugment/last-model-0-int-%s.pt'%iv,
        map_location="cpu")
    model.load_state_dict(checkpoint_model['model_state_dict'])
    #model.load_state_dict(checkpoint_loss['model_state_dict'])
    train_loss = checkpoint_loss['train_losses']
    val_loss = checkpoint_loss['test_losses']


    rec = model(f)
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
    ax[idx].set_ylim(ymax=1e-3)
    ax[idx].legend()

    ax2[idx].imshow(rec[0,:,:].squeeze().detach().numpy())
    #ax2[idx].set_titĺe('{}'.format(ac_combo))

#tikzplotlib.save(join(home,loc_plot))


loss_dict = {"act":act,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict)

print(loss_dict)
plt.show()
