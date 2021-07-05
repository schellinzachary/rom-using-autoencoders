'''
Results Data Augmentation
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/flow_4D.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Convolutional/DatAug.tex"
loc_chpt = "rom-using-autoencoders/01_Thesis/python/Appendix_B/Parameterstudy/Results_dataug"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
###import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


torch.manual_seed(42)
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device="cpu"

#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)



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
        self.add_module('act_a', nn.SiLU())
        self.add_module('act_c', nn.SiLU())


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
        self.add_module('act_a', nn.SiLU())


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

enc_dict = [
    Encoder_2(),
    Encoder_4()
]
dec_dict = [
    Decoder_2(),
    Decoder_4()
]

best_models = (
    # # 2 Layer
    "model0-epoch1989-val_loss1.413E-05",
    # 4 Layer
    "model1-epoch1988-val_loss1.572E-05"
    )
best_modelslr = (
    # 2 Layer with scheduler
    "model0-sched-epoch1988-val_loss2.190E-05",
    # 4 Layer with scheduler
    "model1-sched-epoch1988-val_loss1.854E-05",
    )

train_losses = []
val_losses = []
l2_losses = []
variable = []
min_idx = []



for exp in [1,"sched"]:
    mid=2
    fig, ax = plt.subplots(2)
    if exp == 1:
        best_models = best_models
        fig.suptitle("Dataaugmentation")
    if exp == "sched":
        best_models = best_modelslr
        fig.suptitle("Dataaugmentation with lr adjustment")

    for idx, (encoder, decoder) in enumerate(zip(enc_dict,dec_dict)):

        best_model = best_models[idx]
        #encoder
        encoder = encoder
        #decoder
        decoder = decoder
        #Autoencoder
        model = Autoencoder(encoder, decoder).to(device)

        checkpoint_model = torch.load(join(home,loc_chpt,
            '{}.pt'.format(best_model)),
        map_location="cpu")
        if exp == 1:
            checkpoint_loss = torch.load(join(home,loc_chpt,
                'last-model-{}.pt'.format(idx)),
            map_location="cpu")
        if exp == "sched":
            checkpoint_loss = torch.load(join(home,loc_chpt,
        'last-model-{}_sched.pt'.format(idx)),
        map_location="cpu") 

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
        variable.append(mid)
        

        ax[idx].semilogy(train_loss,'k''--',label='Train')
        ax[idx].semilogy(val_loss,'k''-',label='Test')
        ax[idx].set_xlabel('Epoch')
        ax[idx].set_ylabel('MSE Loss')
        ax[idx].set_title('Model{} '.format(mid))#Change to 3 for Exp. No. 3
        ax[idx].set_ylim(ymax=1e-3)
        ax[idx].legend()
        mid+=2

#tikzplotlib.save(join(home,loc_plot))


loss_dict = {
    "Model":variable,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "l2_loss": l2_losses,
    "epoch val min": min_idx
    }
loss_dict = pd.DataFrame(loss_dict)
print("Dataaugmentation")
print(loss_dict)
plt.show()




