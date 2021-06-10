import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as tensor

device = 'cpu'


# class Encoder(nn.Module):
#     def __init__(self,level,iv):
#         super(Encoder, self).__init__()
#         self.iv = iv
#         if level == "hy":
#             self.act = nn.ELU()
#             self.hid = 30
#         else:
#             self.act = nn.ReLU()
#             self.hid = 40
#         self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=self.hid))
#         self.add_module('activ_1', self.act)
#         self.add_module('layer_c',nn.Linear(in_features=self.hid, out_features=self.iv))
#         self.add_module('activ_c', self.act)
#     def forward(self, x):
#         for _, method in self.named_children():
#             x = method(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self,level,iv):
#         super(Decoder, self).__init__()
#         self.iv = iv
#         if level == "hy":
#             self.act = nn.ELU()
#             self.hid = 30
#         else:
#             self.act = nn.ReLU()
#             self.hid = 40
#         self.add_module('layer_c',nn.Linear(in_features=self.iv, out_features=self.hid))
#         self.add_module('activ_c', self.act)
#         self.add_module('layer_4', nn.Linear(in_features=self.hid,out_features=40))
#     def forward(self, x):
#         for _, method in self.named_children():
#             x = method(x)
#         return x

class Encoder_rare(nn.Module):
    def __init__(self, iv):
        super(Encoder_rare, self).__init__()
        self.iv = iv
        self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=40))
        self.add_module('activ_1', nn.ReLU())
        self.add_module('layer_c',nn.Linear(in_features=40, out_features=self.iv))
        self.add_module('activ_c', nn.ReLU())
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder_rare(nn.Module):
    def __init__(self,iv):
        super(Decoder_rare, self).__init__()
        self.iv = iv
        self.add_module('layer_c',nn.Linear(in_features=self.iv, out_features=40))
        self.add_module('activ_c', nn.ReLU())
        self.add_module('layer_4', nn.Linear(in_features=40,out_features=40))
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Encoder_hy(nn.Module):
    def __init__(self, iv):
        super(Encoder_hy, self).__init__()
        self.iv = iv
        self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=30))
        self.add_module('activ_1', nn.ELU())
        self.add_module('layer_c',nn.Linear(in_features=30, out_features=self.iv))
        self.add_module('activ_c', nn.ELU())
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder_hy(nn.Module):
    def __init__(self,iv):
        super(Decoder_hy, self).__init__()
        self.iv = iv
        self.add_module('layer_c',nn.Linear(in_features=self.iv, out_features=30))
        self.add_module('activ_c', nn.ELU())
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


hydro_models = {
    1 : "int_var-1-epoch4988-val_loss3.797E-07.pt",
    2 : "int_var-2-epoch4988-val_loss3.987E-08.pt",
    3 : "('elu', 'elu')-epoch4989-val_loss4.454E-09.pt",
    4 : "int_var-4-epoch4989-val_loss3.380E-09.pt",
    8 : "int_var-8-epoch4987-val_loss5.104E-10.pt",
    16: "int_var-16-epoch4989-val_loss2.798E-10.pt",
    32: "int_var-32-epoch4988-val_loss3.178E-10.pt"
}

rare_models = {
    1 : "int_var-1-epoch1387-val_loss2.350E-03.pt",
    2 : "int_var-2-epoch4988-val_loss2.051E-07.pt",
    5 : "('relu', 'relu')-epoch4987-val_loss7.191E-09.pt",
    4 : "int_var-4-epoch1-val_loss2.350E-03.pt",
    8 : "int_var-8-epoch4986-val_loss4.791E-07.pt",
    16: "int_var-16-epoch4990-val_loss9.933E-10.pt",
    32: "int_var-32-epoch4990-val_loss1.116E-09.pt"
}

#Load & evaluate models for intrinsic variables variation
def intr_eval(c,iv,level):
    print(iv,level)
    c = tensor(c,dtype=torch.float)
    if level == "hy":
        encoder = Encoder_hy(iv)
        decoder = Decoder_hy(iv)
        checkpoint = torch.load("Models/FullyConnected/Hydro/%s"%hydro_models[iv],
            map_location=torch.device('cpu'))
    else:
        encoder = Encoder_rare(iv)
        decoder = Decoder_rare(iv)
        checkpoint = torch.load("Models/FullyConnected/Rare/%s"%rare_models[iv],
            map_location=torch.device('cpu'))

    model   = Autoencoder(encoder, decoder).to(device)
    if iv == 3 or iv == 5:
        model.load_state_dict(checkpoint['model_state_dict'][0])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    print(hydro_models[iv])
    print(model)

    rec = model(c)
    l2 = torch.norm((c - rec).flatten())/torch.norm(c.flatten()) # calculatre L2-Norm Error

    return l2.detach().numpy()

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy"
c = np.load(join(home,loc_data))
l2 = intr_eval(c,3,"hy")
print(l2)



