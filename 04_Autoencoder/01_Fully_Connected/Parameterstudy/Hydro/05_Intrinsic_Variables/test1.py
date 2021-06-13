import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as tensor

device = 'cpu'


f_hy = np.load('/home/zachary/rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
f_hy = tensor(f_hy,dtype=torch.float)

class Encoder_hy(nn.Module):
    def __init__(self, iv,a,c):
        super(Encoder_hy, self).__init__()
        self.iv = iv
        self.layer_1 = nn.Linear(in_features=40,out_features=30)
        self.activ_1 = nn.ELU()
        self.layer_c = nn.Linear(in_features=30, out_features=self.iv)
        self.activ_c = nn.ELU()
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder_hy(nn.Module):
    def __init__(self,iv,c):
        super(Decoder_hy, self).__init__()
        self.iv = iv
        self.layer_c = nn.Linear(in_features=self.iv, out_features=30)
        self.activ_c = nn.ELU()
        self.layer_4 = nn.Linear(in_features=30,out_features=40)
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


hydro_models = (
    (1 , "int_var-1-epoch4988-val_loss3.797E-07.pt"),
    (2 , "int_var-2-epoch4988-val_loss3.987E-08.pt"),
    (3 , "('elu', 'elu')-epoch4989-val_loss4.454E-09.pt"),
    (4 , "int_var-4-epoch4989-val_loss3.380E-09.pt"),
    (8 , "int_var-8-epoch4987-val_loss5.104E-10.pt"),
    (16, "int_var-16-epoch4989-val_loss2.798E-10.pt"),
    (32, "int_var-32-epoch4988-val_loss3.178E-10.pt")
)

for idx, (iv, model) in enumerate(hydro_models):

    fully = []
    a = nn.ELU()
    c = nn.ELU()
    encoder = Encoder_hy(iv,a,c)
    decoder = Decoder_hy(iv,c)
    checkpoint = torch.load("Results/%s"%model,
        									map_location=torch.device('cpu'))
    model = Autoencoder(encoder, decoder).to(device)
    if iv == 3:
        print(iv)
        model.load_state_dict(checkpoint['model_state_dict'][0])
    else:
        print(iv)
        model.load_state_dict(checkpoint['model_state_dict'])

    rec = model(f_hy)
    l2 = torch.norm((f_hy - rec).flatten())/torch.norm(f_hy.flatten())
    print(iv,l2)
    fully.append(l2)


