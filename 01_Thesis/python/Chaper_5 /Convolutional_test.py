import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import tikzplotlib
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
from __main__ import level
from __main__ import train
from __main__ import iv


device = 'cpu'

# progressbar for training
def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush() 

class params:
    N_EPOCHS = 6000
    BATCH_SIZE = 4
    lr = 1e-4
    if level == "hy":
        LATENT_DIM = 3
    else:
        LATENT_DIM = 5



class net:
    class Encoder(nn.Module):
        def __init__(self):
            super(net.Encoder, self).__init__()
            self.m = nn.ZeroPad2d((0,0,1,1))
            self.convE1 = nn.Conv2d(1,32,(6,10),stride=(3,10))
            self.convE2 = nn.Conv2d(32,64,(4,10),stride=(4,10))
            self.linearE1 = nn.Linear(in_features=256,out_features=params.LATENT_DIM)
            self.add_module('act',nn.SiLU())


        def forward(self, x):
            x = self.m(x)
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self):
            super(net.Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=params.LATENT_DIM, out_features=256)
            self.convD1 = nn.ConvTranspose2d(64,32,(4,10),stride=(4,10))
            self.convD2 = nn.ConvTranspose2d(32,1,(4,10),stride=(3,10))
            self.add_module('act',nn.SiLU())


        def forward(self, x):
            x = self.linearD1(x)
            dim = x.shape[0]
            x = torch.reshape(x,[dim,64,2,2])
            x = self.act(self.convD1(x))
            x = self.act(self.convD2(x))
            return x



    class Autoencoder(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, x):
            z = self.enc(x)
            x = self.dec(z)
            return x, z


#INIT Model, Decoder and Encoder
################################
encoder = net.Encoder()
decoder = net.Decoder()
model = net.Autoencoder(encoder, decoder).to(device)

#Load model
###########
checkpoint = torch.load('State_Dict/Conv_{}.pt'.format(level))
model.load_state_dict(checkpoint['model_state_dict'])
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']
N_EPOCHS = checkpoint['epoch']



#Train Model with different # of intrinsic variables
####################################################

def model_train(c,iv,level):
    device = "cuda"
    params.LATENT_DIM = iv
    #INIT Model, Decoder and Encoder
    encoder = net.Encoder()
    decoder = net.Decoder()
    model = net.Autoencoder(encoder, decoder).to(device)
    optimizer = Adam(params=model.parameters(), lr=params.lr)
    loss_crit = nn.MSELoss()
    train_losses = []
    val_losses = []
    if level == "rare":
        train_in = c[0:7 and 16:39]
        val_in = c[8:15]
    else:
        train_in = c[0:31]
        val_in = c[32:39]
    train_iterator = DataLoader(train_in, batch_size = params.BATCH_SIZE)
    test_iterator = DataLoader(val_in, batch_size = int(len(c)*0.2))
    def train():
        model.train()
        train_loss = 0.0
        for batch_ndx, x in enumerate(train_iterator):
            x = x.to(device)
            optimizer.zero_grad()
            predicted,z = model(x)
            loss = loss_crit(predicted,x)
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
                predicted,z = model(x)
                loss = loss_crit(predicted,x)
                test_loss += loss.item()
        return test_loss
    test_losses = []
    val_losses = []
    for epoch in range(params.N_EPOCHS):
        #train the model
        train_loss = train()
        test_loss = test()
        #save the loss
        train_loss /= len(train_iterator)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        progressBar(epoch,params.N_EPOCHS)
    print('the level is',level)
    torch.save({
    'epoch': epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses':train_losses,
    'test_losses': test_losses
    },'Results/Conv{}_{}.pt'.format(iv,level))
    return
