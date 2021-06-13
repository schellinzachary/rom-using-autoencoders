import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as tensor


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
        x = self.act_c(self.linearE1(x))
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
    1 : "model0-int1-epoch1554-val_loss2.010E-04.pt",
    2 : "model0-int2-epoch1990-val_loss7.078E-05.pt",
    4 : "model0-int4-epoch1734-val_loss6.353E-06.pt",
    8 : "model0-int8-epoch1752-val_loss6.601E-06.pt",
    16: "model0-int16-epoch1661-val_loss6.285E-06.pt",
    32: "model0-int32-epoch1741-val_loss6.148E-06.pt"
}

#Load & evaluate models for intrinsic variables variation
def intr_eval(c,iv,level):
    c = tensor(c,dtype=torch.float)
    encoder = Encoder_2(iv)
    decoder = Decoder_2(iv)
    model   = Autoencoder(encoder, decoder).to(device)
    
    checkpoint = torch.load("Models/Convolutional/%s"%models[iv],
            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    rec = model(c)
    l2 = torch.norm((c - rec).flatten())/torch.norm(c.flatten()) # calculatre L2-Norm Error
    return(l2.detach().numpy())

