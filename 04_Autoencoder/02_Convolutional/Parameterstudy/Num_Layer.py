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

#best fold
Train_fold = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
Test_fold = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

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


BATCH_SIZE = 2
lr = 1e-4
N_EPOCHS = 10

class Encoder_3(nn.Module):
    def __init__(self):
        super(Encoder_3, self).__init__()
        self.convE1 = nn.Conv2d(1,4,(3,3),stride=(3,3),padding=(1,1))
        self.convE2 = nn.Conv2d(4,8,(3,3),stride=(3,3),padding=(0,1))
        self.convE3 = nn.Conv2d(8,16,(3,3),stride=(3,3),padding=(0,1))
        self.linearE1 = nn.Linear(in_features=128,out_features=5)
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
        self.linearD1 = nn.Linear(in_features=5, out_features=128)
        self.convD1 = nn.ConvTranspose2d(16,8,(3,3),stride=(3,3),padding=(0,1))
        self.convD2 = nn.ConvTranspose2d(8,4,(3,3),stride=(3,3),padding=(0,1))
        self.convD3 = nn.ConvTranspose2d(4,1,(3,3),stride=(3,3),padding=(1,1))
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,8])
        x = self.act(self.convD1(x))
        print(x.shape)
        x = self.act(self.convD2(x))
        print(x.shape)
        x = self.act(self.convD3(x))
        print(x.shape)
        return x

class Encoder_4(nn.Module):
    def __init__(self):
        super(Encoder_4, self).__init__()
        self.convE1 = nn.Conv2d(1,2,(3,3),stride=(3,3))
        self.convE2 = nn.Conv2d(2,4,(3,3),stride=(3,3))
        self.convE3 = nn.Conv2d(4,8,(3,3),stride=(3,3))
        self.convE4 = nn.Conv2d(8,16,(4,4),stride=(4,4))
        self.linearE1 = nn.Linear(in_features=128,out_features=5)
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
        self.linearD1 = nn.Linear(in_features=5, out_features=128)
        self.convD1 = nn.ConvTranspose2d(16,8,(4,4),stride=(4,4))
        self.convD2 = nn.ConvTranspose2d(8,4,(3,3),stride=(3,3))
        self.convD3 = nn.ConvTranspose2d(4,2,(3,3),stride=(3,3))
        self.convD4 = nn.ConvTranspose2d(2,1,(3,3),stride=(3,3))
        self.add_module('act',nn.ReLU())


    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,1,8])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        x = self.act(self.convD3(x))
        x = self.act(self.convD4(x))
        return x

enc_dict = {
    Encoder_3(),
    Encoder_4()
}
dec_dict = {
    Decoder_4(),
    Decoder_3()
}

for idx, (encoder, decoder) in enumerate(zip(enc_dict,dec_dict)):
    train_in = f[Train_fold]
    val_in = f[Test_fold]

    train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
    test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))



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
    encoder = encoder


    #decoder
    decoder = decoder

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
    #check the model
    print(model)

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
  
    #save top 3 models
    # for i in range(3):
    #     k_models = np.array(k_models)
    #     torch.save({
    # 'epoch': k_models[i,1],
    # 'model_state_dict':k_models[i,2],
    # },'Results_layer/fold{}-epoch{}-val_loss{:.3E}s.pt'.format(idx,
    #     k_models[i,1],
    #     k_models[i,0]))
    # #save last model
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict':model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'train_losses':train_losses,
    #     'test_losses': test_losses
    #     },'Results_layer/last-fold-{}.pt'.format(idx))





