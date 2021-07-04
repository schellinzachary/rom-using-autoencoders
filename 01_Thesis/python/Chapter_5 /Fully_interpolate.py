import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as tensor

device = 'cpu'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z = self.enc(x)
        predicted = self.dec(z)
        return predicted, z

class Encoder(nn.Module):
    def __init__(self, act_h, act_c, hid, code):
        super(Encoder, self).__init__()
        self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=hid))
        self.add_module('activ_1', act_h)
        self.add_module('layer_c',nn.Linear(in_features=hid, out_features=code))
        self.add_module('activ_c', act_c)
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder(nn.Module):
    def __init__(self, act_h, hid, code):
        super(Decoder, self).__init__()
        self.add_module('layer_c',nn.Linear(in_features=code, out_features=hid))
        self.add_module('activ_c', act_h)
        self.add_module('layer_4', nn.Linear(in_features=hid,out_features=40))
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

hydro_models = {
    2: "del_var-2-epoch4990-val_loss2.521E-08.pt",
    3: "del_var-3-epoch4988-val_loss2.909E-08.pt",
    4: "del_var-4-epoch4990-val_loss2.598E-08.pt",
    5: "del_var-5-epoch4990-val_loss1.776E-07.pt"
}

rare_models = {
    2: "del_var-2-epoch4988-val_loss2.904E-07.pt",
    3: "del_var-3-epoch3000-val_loss9.540E-08.pt",
    4: "del_var-4-epoch4988-val_loss1.631E-07.pt",
    5: "del_var-5-epoch4989-val_loss1.623E-07.pt"
}

#set variables
activations = {
    'relu': nn.ReLU(),
    'elu': nn.ELU()
}


#Load & evaluate models for intrinsic variables variation
def fully_int(del_var,c,level):
    c = tensor(c, dtype=torch.float)


    if level == "hy":
        # act_h = activations["elu"]
        # act_c = activations["elu"]
        encoder = Encoder(nn.ELU(), nn.ELU(), 30, 3)
        decoder = Decoder(nn.ELU(), 30, 3)
        checkpoint = torch.load("Models/FullyConnected/Hydro/%s"%hydro_models[del_var],
            map_location=torch.device('cpu'))
    if level == "rare":

        # act_h = activations["relu"]
        # act_c = activations["relu"]
        encoder = Encoder(nn.ReLU(), nn.ReLU(), 40, 5)
        decoder = Decoder(nn.ReLU(), 40, 5)
        checkpoint = torch.load("Models/FullyConnected/Rare/%s"%rare_models[del_var],
            map_location=torch.device('cpu'))

    model   = Autoencoder(encoder, decoder).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])


    rec, z = model(c)
    paramcount = count_parameters(model)
    #print(paramcount)
    #r2 = 1 - (torch.sum((c - rec))**2 / torch.sum((rec - torch.mean(c))**2))
    #print(r2)
    return rec, z

def decoder_int(del_var,c, level):
    c = tensor(c, dtype=torch.float)

    if level == "hy":
        #act_h = activations["elu"]
        decoder = Decoder(nn.ELU(), 30, 3)
        checkpoint = torch.load("Models/FullyConnected/Hydro/%s"%hydro_models[del_var],
            map_location=torch.device('cpu'))

    if level == "rare":
        #act_h = activations["relu"]
        decoder = Decoder(nn.ReLU(), 40, 5)
        checkpoint = torch.load("Models/FullyConnected/Rare/%s"%rare_models[del_var],
            map_location=torch.device('cpu'))

    state_dict = checkpoint['model_state_dict']
    with torch.no_grad():
        decoder.layer_c.weight.copy_(state_dict['dec.layer_c.weight'])
        decoder.layer_c.bias.copy_(state_dict['dec.layer_c.bias'])
        decoder.layer_4.weight.copy_(state_dict['dec.layer_4.weight'])
        decoder.layer_4.bias.copy_(state_dict['dec.layer_4.bias'])

    rec = decoder(c)

    return rec








