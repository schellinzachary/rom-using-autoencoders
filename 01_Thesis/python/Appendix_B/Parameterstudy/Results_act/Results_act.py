'''
Different Activations
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/flow_4D.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Convolutional/4l_activations.tex"


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
    def __init__(self, a=None, c=None):
        super(Encoder_2, self).__init__()
        self.convE1 = nn.Conv2d(
            1,8,(5,5),
            stride=(5,5)
            )
        self.convE2 = nn.Conv2d(
            8,16,(5,5),
            stride=(5,5)
            )
        self.linearE1 = nn.Linear(in_features=128,out_features=5)
        self.add_module('act_a', a)
        self.add_module('act_c', c)


    def forward(self, x):
        x = self.act_a(self.convE1(x))
        x = self.act_a(self.convE2(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.act_a(self.linearE1(x))
        return x

class Decoder_2(nn.Module):
    def __init__(self, a=None):
        super(Decoder_2, self).__init__()
        self.linearD1 = nn.Linear(
            in_features=5,
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
        self.add_module('act_a', a)




    def forward(self, x):
        x = self.act_a(self.linearD1(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,8])
        x = self.act_a(self.convD1(x))
        x = self.convD2(x)
        return x

class Encoder_3(nn.Module):
    def __init__(self,a, c):
        super(Encoder_3, self).__init__()
        self.convE1 = nn.Conv2d(
            1,8,(3,3),
            stride=(3,3),
            padding=(1,1)
            )
        self.convE2 = nn.Conv2d(
            8,16,(3,3),
            stride=(3,3),
            padding=(0,1)
            )
        self.convE3 = nn.Conv2d(
            16,32,(3,3),
            stride=(3,3),
            padding=(0,1)
            )
        self.linearE1 = nn.Linear(in_features=256,
            out_features=5)
        self.add_module('act_a', a)
        self.add_module('act_c', c)


    def forward(self, x):
        x = self.act_a(self.convE1(x))
        x = self.act_a(self.convE2(x))
        x = self.act_a(self.convE3(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.act_c(self.linearE1(x))
        return x

class Decoder_3(nn.Module):
    def __init__(self, a):
        super(Decoder_3, self).__init__()
        self.linearD1 = nn.Linear(in_features=5,
            out_features=256
            )
        self.convD1 = nn.ConvTranspose2d(
            32,16,(3,3),
            stride=(3,3),
            padding=(0,1),
            output_padding=(0,1)
            )
        self.convD2 = nn.ConvTranspose2d(
            16,8,(3,3),
            stride=(3,3),
            padding=(0,1),
            output_padding=(0,1)
            )
        self.convD3 = nn.ConvTranspose2d(
            8,1,(3,3),
            stride=(3,3),
            padding=(1,2)
            )
        self.add_module('act_a', a)


    def forward(self, x):
        x = self.act_a(self.linearD1(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,32,1,8])
        x = self.act_a(self.act(self.convD1(x)))
        x = self.act_a(self.act(self.convD2(x)))
        x = self.convD3(x)
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


#set variables
activations = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'silu': nn.SiLU(),
    'tanh': nn.Tanh(),
    'leaky': nn.LeakyReLU() 
}

#variable combinations
experiments = (
    ('elu','elu'),
    ('elu','silu'),
    ('elu','tanh'),
    ('leaky','leaky'),
    ('leaky','tanh'),
    ('relu','relu'),
    ('silu','silu'),
    ('tanh','tanh')
    )


best_models = (
	#uncomment for models for 2 Layer Autoencoder
    "model0-act-('elu', 'elu')-epoch1969-val_loss6.500E-06",
    "model0-act-('elu', 'silu')-epoch1974-val_loss5.696E-06",
    "model0-act-('elu', 'tanh')-epoch1970-val_loss5.939E-06",
    "model0-act-('leaky', 'leaky')-epoch1976-val_loss1.093E-05",
    "model0-act-('leaky', 'tanh')-epoch1974-val_loss6.591E-06",
    "model0-act-('relu', 'relu')-epoch1984-val_loss1.319E-05",
    "model0-act-('silu', 'silu')-epoch1972-val_loss7.856E-06",
    "model0-act-('tanh', 'tanh')-epoch1909-val_loss7.762E-06"
    #uncomment models for 4 Layer Autoencoder
    # "model1-act-('elu', 'elu')-epoch1365-val_loss6.916E-06",
    # "model1-act-('elu', 'silu')-epoch1808-val_loss7.532E-06",
    # "model1-act-('elu', 'tanh')-epoch1498-val_loss9.252E-06",
    # "model1-act-('leaky', 'leaky')-epoch1971-val_loss9.615E-06",
    # "model1-act-('leaky', 'tanh')-epoch1722-val_loss9.401E-06",
    # "model1-act-('relu', 'relu')-epoch1985-val_loss1.073E-05",
    # "model1-act-('silu', 'silu')-epoch1550-val_loss6.497E-06",
    # "model1-act-('tanh', 'tanh')-epoch975-val_loss8.390E-06"
    )

#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)


train_losses = []
val_losses = []
l2_losses = []
act = []
min_idx = []

fig, ax = plt.subplots(8,1)

for idx, (ac_combo, best_model) in enumerate(zip(experiments,best_models)):
    
    idx_model = 0 #0 for 2 Layer and 1 for 4 Layer

    a, c = ac_combo
    a = activations[a]
    c = activations[c]
    #encoder
    encoder = Encoder_2(a,c) #change to Encoder_4(a,c) for 4 layer

    #decoder
    decoder = Decoder_2(a) #change to Decoder_4(a) for 4 layer

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    checkpoint_model = torch.load('{}.pt'.format(best_model),map_location=torch.device('cpu'))
    checkpoint_loss = torch.load('last-model-{}-act-{}.pt'.format(idx_model,
    	ac_combo
    	),
    map_location=torch.device('cpu'))
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
    act.append(ac_combo)
    
    ax[idx].semilogy(train_loss,'k''--',label='Train')
    ax[idx].semilogy(val_loss,'k''-',label='Test')
    ax[idx].set_xlabel('Epoch')
    ax[idx].set_ylabel('MSE Loss')
    ax[idx].set_title('{} '.format(ac_combo))
    ax[idx].set_ylim(ymax=1e-3)
    ax[idx].legend()

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
