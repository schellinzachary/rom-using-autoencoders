'''
K-Fold to find best fold
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/flow_4D.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Convolutional/Batch.tex"


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
#import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)



class Encoder_2(nn.Module):
    def __init__(self):
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
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.act(self.convE1(x))
        x = self.act(self.convE2(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.linearE1(x)
        return x

class Decoder_2(nn.Module):
    def __init__(self):
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
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,8])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        return x

class Encoder_3(nn.Module):
    def __init__(self):
        super(Encoder_3, self).__init__()
        self.convE1 = nn.Conv2d(
            1,4,(3,3),
            stride=(3,3),
            padding=(1,1)
            )
        self.convE2 = nn.Conv2d(
            4,8,(3,3),
            stride=(3,3),
            padding=(0,1)
            )
        self.convE3 = nn.Conv2d(
            8,16,(3,3),
            stride=(3,3),
            padding=(0,1)
            )
        self.linearE1 = nn.Linear(in_features=128,
            out_features=5)
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.act(self.convE1(x))
        print(x.shape)
        x = self.act(self.convE2(x))
        print(x.shape)
        x = self.act(self.convE3(x))
        print(x.shape)
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.linearE1(x)
        return x

class Decoder_3(nn.Module):
    def __init__(self):
        super(Decoder_3, self).__init__()
        self.linearD1 = nn.Linear(in_features=5,
            out_features=128
            )
        self.convD1 = nn.ConvTranspose2d(
            16,8,(3,3),
            stride=(3,3),
            padding=(0,1),
            output_padding=(0,1)
            )
        self.convD2 = nn.ConvTranspose2d(
            8,4,(3,3),
            stride=(3,3),
            padding=(0,1),
            output_padding=(0,1)
            )
        self.convD3 = nn.ConvTranspose2d(
            4,1,(3,3),
            stride=(3,3),
            padding=(1,2)
            )
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,8])
        x = self.act(self.convD1(x))
        print(x.shape)
        x = self.act(self.convD2(x))
        print(x.shape)
        x = self.convD3(x)
        return x

class Encoder_4(nn.Module):
    def __init__(self):
        super(Encoder_4, self).__init__()
        self.convE1 = nn.Conv2d(
            1,2,(3,3),
            stride=(3,3),
            padding=(3,2)
            )
        self.convE2 = nn.Conv2d(
            2,4,(3,3),
            stride=(3,3),
            padding=(3,2),
            )
        self.convE3 = nn.Conv2d(
            4,8,(3,3),
            stride=(3,3),
            padding=(3,2)
            )
        self.convE4 = nn.Conv2d(
            8,16,(3,3),
            stride=(3,3),
            padding=(0,0)
            )
        self.linearE1 = nn.Linear(in_features=48,
            out_features=5
            )
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.act(self.convE1(x))
        x = self.act(self.convE2(x))
        x = self.act(self.convE3(x))
        x = self.act(self.convE4(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.linearE1(x)
        return x

class Decoder_4(nn.Module):
    def __init__(self):
        super(Decoder_4, self).__init__()
        self.linearD1 = nn.Linear(in_features=5,
        out_features=48
        )
        self.convD1 = nn.ConvTranspose2d(
            16,8,(3,3),
            stride=(3,3),
            padding=(0,0),
            output_padding=(0,0)
            )
        self.convD2 = nn.ConvTranspose2d(
            8,4,(3,3),
            stride=(3,3),
            padding=(2,2),
            output_padding=(0,1)
            )
        self.convD3 = nn.ConvTranspose2d(
            4,2,(3,3),
            stride=(3,3),
            padding=(3,2),
            output_padding=(0,0)
            )
        self.convD4 = nn.ConvTranspose2d(
            2,1,(3,3),
            stride=(3,3),
            padding=(1,2)
            )
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,3])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        x = self.act(self.convD3(x))
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

enc_dict = [
    Encoder_2(),#only for Exp. No. 3
    Encoder_3(),
    Encoder_4()
]
dec_dict = [
    Decoder_2(),#only for Exp. No. 3
    Decoder_3(),
    Decoder_4()
]

best_models = (
    # scaled data not working
    # "model3-epoch1399-val_loss7.762E-03-exp-1",
    # "model4-epoch1094-val_loss1.474E-01-exp-1",
    # non scaled working
    # "model3-epoch1975-val_loss1.537E-05-exp-2",
    # "model4-epoch1924-val_loss8.178E-04-exp-2"
    # increased batch size    
    "model2-epoch1969-val_loss8.009E-06-exp-3",
    "model3-epoch1980-val_loss1.326E-05-exp-3",
    "model4-epoch109-val_loss6.976E-03-exp-3"     
    )

train_losses = []
val_losses = []
l2_losses = []
variable = []
min_idx = []



fig, ax = plt.subplots(1,3)#Change to 3 for Exp. No. 3

for idx, (encoder, decoder) in enumerate(zip(enc_dict,dec_dict)):

    best_model = best_models[idx]
    #encoder
    encoder = encoder
    #decoder
    decoder = decoder
    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    checkpoint_model = torch.load('{}.pt'.format(best_model))
    checkpoint_loss = torch.load('last-model-{}exp-3.pt'.format(idx+2))#Change "exp-" to 3 for Exp. No. 3
                                                                       #and "idx+" to 2 for Exp. No. 3                             

    model.load_state_dict(checkpoint_model['model_state_dict'])
    train_loss = checkpoint_loss['train_losses']
    val_loss = checkpoint_loss['test_losses']


    rec = model(f)

    l2_loss = torch.norm((f - rec).flatten())/torch.norm(f.flatten())
    l2_loss = l2_loss.to('cpu')

    train_losses.append(np.min(train_loss))
    val_losses.append(np.min(val_loss))
    l2_losses.append(l2_loss.detach().numpy())
    min_idx.append(val_loss.index(min(val_loss)))
    variable.append(idx+2)
    

    ax[idx].semilogy(train_loss,'k''--',label='Train')
    ax[idx].semilogy(val_loss,'k''-',label='Test')
    ax[idx].set_xlabel('Epoch')
    ax[idx].set_ylabel('MSE Loss')
    ax[idx].set_title('Model{} '.format(idx+3))#Change to 3 for Exp. No. 3
    ax[idx].set_ylim(ymax=1e-2)
    ax[idx].legend()

#tikzplotlib.save(join(home,loc_plot))


loss_dict = {
    "Model":variable,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict)

print(loss_dict)
plt.show()



