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

def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush() # progressbar for training

class params:
    N_EPOCHS = 2000
    BATCH_SIZE = 16
    lr = 1e-4
    if level == "hy":
        LATENT_DIM = 3
    else:
        LATENT_DIM = 5


class net:

    class Encoder(nn.Module):
        def __init__(self):
            super(net.Encoder, self).__init__()
            self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=40))
            self.add_module('activ_1', nn.LeakyReLU())
            self.add_module('layer_c',nn.Linear(in_features=40, out_features=params.LATENT_DIM))
            self.add_module('activ_c', nn.Tanh())
        def forward(self, x):
            for _, method in self.named_children():
                x = method(x)
            return x



    class Decoder(nn.Module):
        def __init__(self):
            super(net.Decoder, self).__init__()
            self.add_module('layer_c',nn.Linear(in_features=params.LATENT_DIM, out_features=40))
            self.add_module('activ_c', nn.LeakyReLU())
            self.add_module('layer_4', nn.Linear(in_features=40,out_features=40))
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
            return predicted, z

#INIT Model, Decoder and Encoder
################################
encoder = net.Encoder()
decoder = net.Decoder()
model = net.Autoencoder(encoder, decoder).to(device)

#Load model best performing model from Parameterstudy
#####################################################
encoder = net.Encoder()
decoder = net.Decoder()
model = net.Autoencoder(encoder, decoder).to(device)
checkpoint = torch.load('State_Dict/Fully_{}.pt'.format(level)) #best model from Parameterstudy
#checkpoint = torch.load('Results/Fully8_{}.pt'.format(level)) # best model from intrinsic code variation
model.load_state_dict(checkpoint['model_state_dict'])
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']
N_EPOCHS = checkpoint['epoch']

#Load & evaluate models from intrinsic variables variation
##########################################################
def intr_eval(c,iv,level):
    params.LATENT_DIM = iv
    encoder = net.Encoder()
    decoder = net.Decoder()
    model = net.Autoencoder(encoder, decoder).to(device)
    checkpoint = torch.load('Results/Fully{}_{}.pt'.format(iv,level))
    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']
    rec,code = model(c)
    l2 = torch.norm((c - rec).flatten())/torch.norm(c.flatten()) # calculatre L2-Norm Error
    # plt.semilogy(np.linspace(0,N_EPOCHS,N_EPOCHS+1),train_losses)
    # plt.semilogy(np.linspace(0,N_EPOCHS,N_EPOCHS+1),test_losses)
    # plt.title('{}'.format(iv))
    # plt.show()
    return(l2.detach().numpy())





#Train Model with different # of intrinsic variables
####################################################

def model_train(c,iv,level):
    device = "cpu"
    params.LATENT_DIM = iv
    #INIT Model, Decoder and Encoder
    encoder = net.Encoder()
    decoder = net.Decoder()
    model = net.Autoencoder(encoder, decoder).to(device)
    optimizer = Adam(params=model.parameters(), lr=params.lr)
    loss_crit = nn.MSELoss()
    train_losses = []
    val_losses = []
    train_in = c[0:3999]
    val_in = c[4000:4999]
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
    torch.save({
    'epoch': epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses':train_losses,
    'test_losses': test_losses
    },'Results/Fully{}_{}.pt'.format(iv,level))
    return
