import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as tensor

device = 'cpu'


class Encoder(nn.Module):
    def __init__(self,level,iv):
        super(Encoder, self).__init__()
        self.iv = iv
        if level == "hy":
            self.act = nn.ELU()
            self.hid = 30
        else:
            self.act = nn.ReLU()
            self.hid = 40
        self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=self.hid))
        self.add_module('activ_1', self.act)
        self.add_module('layer_c',nn.Linear(in_features=self.hid, out_features=self.iv))
        self.add_module('activ_c', self.act)
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder(nn.Module):
    def __init__(self,level,iv):
        super(Decoder, self).__init__()
        self.iv = iv
        if level == "hy":
            self.act = nn.ELU()
            self.hid = 30
        else:
            self.act = nn.ReLU()
            self.hid = 40
        self.add_module('layer_c',nn.Linear(in_features=self.iv, out_features=self.hid))
        self.add_module('activ_c', self.act)
        self.add_module('layer_4', nn.Linear(in_features=self.hid,out_features=40))
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


#Load & evaluate models for intrinsic variables variation
def intr_eval(c,iv,level):
    c = tensor(c,dtype=torch.float)
    encoder = Encoder(level,iv)
    decoder = Decoder(level,iv)
    model   = Autoencoder(encoder, decoder).to(device)

    if level == "hy":
        checkpoint = torch.load("Models/('elu', 'elu')-epoch4989-val_loss4.454E-09.pt",
            map_location=torch.device('cpu'))
    else: 
        checkpoint = torch.load("Models/('relu', 'relu')-epoch4987-val_loss7.191E-09",
            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'][0])

    rec,code = model(c)
    l2 = torch.norm((c - rec).flatten())/torch.norm(c.flatten()) # calculatre L2-Norm Error
    return(l2.detach().numpy())
