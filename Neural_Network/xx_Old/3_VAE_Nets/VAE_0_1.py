'''
VAE 0.1
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from random import randint
import sys
import os




device = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 32     
N_EPOCHS = 500      # times to run the model on complete data
INPUT_DIM = 40 # size of each input
HIDDEN_DIM = 20    # hidden dimension
LATENT_DIM = 5     # latent vector dimension
beta = 1
lr = 1e-3           # learning rate



#Load Data and Shuffle
f = np.load('preprocessed_samples_lin_substract50.npy')
f = tensor(f, dtype=torch.float).to(device)
np.random.shuffle(f)
train_in = f[0:2999]
val_in = f[3000:3749]  


train_iterator = DataLoader(train_in, batch_size=BATCH_SIZE)
test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))




# Defining neural network
# Encoder



class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, lat_dim):
        # initialize as nn.Module
        super().__init__()
        self.linear0 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear1 = nn.Linear(in_features=hidden_dim, out_features=2*lat_dim)
        self.linear21 = nn.Linear(in_features=2 * lat_dim, out_features=lat_dim) #mu layer
        self.linear22 = nn.Linear(in_features=2* lat_dim, out_features=lat_dim) #logvariance layer
        self.activation_out = Swish()

    def forward(self, x):
        x = self.activation_out(self.linear0(x))
        x = self.activation_out(self.linear1(x))
        x21 = self.linear21(x)
        x22 = self.linear22(x)

        return x21,x22

# Decoder


class Decoder(nn.Module):
    def __init__(self, lat_dim, hidden_dim, output_dim):

        # initialize as nn.Module
        super().__init__()
        self.linear3 = nn.Linear(in_features=lat_dim, out_features=hidden_dim)
        self.linear4 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.activation_out = Swish()
        self.activation_out1 = nn.Sigmoid()

    def forward(self, x):
        x = self.activation_out(self.linear3(x))
        x = self.activation_out1(self.linear4(x))

        return x

class VAE(nn.Module):
    '''
    Autoencoder which takes the encoder and the decoder
    '''
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec


    def forward(self, x):
        #encoder
        mu,logvar = self.enc(x)


        #sample from distribution & reparametrize
        std = torch.exp(logvar*0.5)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        #decode
        predicted = self.dec(z)
        return predicted, mu, logvar

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



#encoder
encoder = Encoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#decoder
decoder = Decoder(LATENT_DIM,HIDDEN_DIM,INPUT_DIM)

#VAE
model = VAE(encoder, decoder).to(device)

#optimizer
optimizer = Adam(params=model.parameters(), lr=lr)

loss_crit = nn.L1Loss()
train_losses = []
test_losses = []

def train():

    model.train()

    train_loss = 0.0

    for batch_ndx, x in enumerate(train_iterator):

        x = x.to(device)

        #update the gradients to zero
        optimizer.zero_grad()

        #forward pass
        predicted, mu, logvar = model(x)

        #reconsruction Loss
        recon_loss = loss_crit(predicted, x)

        #kl Loss
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)

        #total Loss
        loss = recon_loss + beta * kl_loss

        #backward pass
        loss.backward()
        train_loss += loss.item()

        #update weights
        optimizer.step()

    return train_loss, kl_loss, recon_loss

def test():

    #set the evaluation mode
    model.eval()

    #test loss for the data
    test_loss = 0

    with torch.no_grad():
        for i, x in enumerate(test_iterator):

            x = x.to(device)

            #forward pass

            z, mu, logvar = model(x)

            #reconstruction Loss
            recon_loss = loss_crit(z,x)

            #kl Loss
            kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 -1.0 -logvar)

            #total loss
            loss = recon_loss + kl_loss
            test_loss += loss.item()

        return test_loss



for n_iter in range(N_EPOCHS):

    
    #train and evaluate the model
    train_loss, kl_loss, recon_loss = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(train_in)
    #kl_loss /=len(train_in)
    #recon_loss /=len(train_in)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {n_iter}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f},KL Loss: {kl_loss:.5f}')

    if n_iter % 100 == 0:

     i = randint(0,748)
     x = val_in[i].to(device)

     predicted,mu, logvar = model(x)
     x = x.to('cpu')
     predicted = predicted.to('cpu')
     data = x.detach().numpy()
     predict = predicted.detach().numpy()
    
     plt.plot(x, label='Original')
     plt.plot(predict, label='Predicted')
     plt.legend()
     plt.show()


#save the models state dictionary for inference
model.eval()
torch.save(model.state_dict(),'VAE_0_1_STATE_DICT_BETA_10.pt') 
