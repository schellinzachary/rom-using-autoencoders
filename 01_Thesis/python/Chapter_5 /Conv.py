import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

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

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z = self.enc(x)
        predicted = self.dec(z)
        return predicted

activations = {
    'silu': nn.SiLU(),
    'elu': nn.ELU()
    }

#Load & evaluate models for intrinsic variables variation
def conv(c):
    c = tensor(c,dtype=torch.float)

    act_h = activations["elu"]
    act_c = activations["silu"] 
    encoder = Encoder_2(act_h, act_c)
    decoder = Decoder_2(act_h)
    model   = Autoencoder(encoder, decoder).to(device)
    
    checkpoint = torch.load("Models/Convolutional/model0-act-('elu', 'silu')-epoch1974-val_loss5.696E-06.pt",
            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    rec = model(c)

    return rec 

