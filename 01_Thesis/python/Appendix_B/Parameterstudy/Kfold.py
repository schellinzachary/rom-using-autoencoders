'''
K-Fold to find best fold
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/flow_4D.npy"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import tikzplotlib


import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.model_selection import KFold

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(k_models,ac_combo):
    k_models = np.array(k_models)
    k_models = k_models[k_models[:,0].argsort()]
    return np.ndarray.tolist(k_models[:3])

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

#load data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float).to(device)

kf = KFold(n_splits=5)

BATCH_SIZE = 2
lr = 1e-4
N_EPOCHS = 1000

for idx, (train, test) in enumerate(kf.split(f)):

    print("Train:", train)
    print("TEST:", test)
    
    continue

    train_in = f[train]
    val_in = f[test]

    train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
    test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.convE1 = nn.Conv2d(1,8,(5,5),stride=(5,5))
            self.convE2 = nn.Conv2d(8,16,(5,5),stride=(5,5))
            self.linearE1 = nn.Linear(in_features=128,out_features=5)
            self.add_module('act',nn.ReLU())


        def forward(self, x):
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            x = self.linearE1(x)
            return x

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=5, out_features=128)
            self.convD1 = nn.ConvTranspose2d(16,8,(5,5),stride=(5,5))
            self.convD2 = nn.ConvTranspose2d(8,1,(5,5),stride=(5,5))
            self.add_module('act',nn.ReLU())


        def forward(self, x):
            x = self.linearD1(x)
            dim = x.shape[0]
            x = torch.reshape(x,[dim,16,1,8])
            x = self.act(self.convD1(x))
            x = self.act(self.convD2(x))
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

    #encoder
    encoder = Encoder()

    #decoder
    decoder = Decoder()

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    decoder.apply(weight_reset)
    encoder.apply(weight_reset)

    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1)

    loss_crit = nn.MSELoss()
    train_losses = []
    val_losses = []


    def train():

        model.train()

        train_loss = 0.0

        for batch_ndx, x in enumerate(train_iterator):

            x = x.to(device)

            optimizer.zero_grad()

            predicted = model(x)

            loss = loss_crit(x,predicted)

            loss.backward()
            train_loss += loss.item()

            optimizer.step()

        return train_loss

    def test():

        model.eval()

        test_loss = 0

        with torch.no_grad():
            for i, x in enumerate(test_iterator):

                x = x.to(device)

                predicted = model(x)

                loss = loss_crit(x,predicted)
                test_loss += loss.item()

            return test_loss


    train_losses = []
    test_losses = []
    k_models = []


    for epoch in tqdm(range(N_EPOCHS)):
        train_loss = train()
        test_loss = test()

        train_loss /= len(train_iterator)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step()

        k_models.append((
            test_loss,
            epoch,
            model.state_dict())
        )

        if (epoch > 1) and(epoch % 10 == 0):
            k_models = save_checkpoint(k_models,idx)
  
    #save top 3 models4
    for i in range(3):
        k_models = np.array(k_models)
        torch.save({
    'epoch': k_models[i,1],
    'model_state_dict':k_models[i,2],
    },'Results/fold{}-epoch{}-val_loss{:.3E}s.pt'.format(idx,
        k_models[i,1],
        k_models[i,0]))
    #save last model
    torch.save({
        'epoch': epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses':train_losses,
        'test_losses': test_losses
        },'Results/last-fold-{}.pt'.format(idx))



