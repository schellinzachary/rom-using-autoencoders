'''
Convolutional Autoencoder v1.2
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io as sio


device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_EPOCHS = 6000
BATCH_SIZE = 2
INPUT_DIM = 25*200
HIDDEN_DIM = 500
LATENT_DIM = 5
lr = 1e-5



#load data
f1 = np.load('preprocessed_samples_conv.npy')
f1 = tensor(f1, dtype=torch.float).to(device)
train_in = f1[0:19]
val_in = f1[20:24]
train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
test_iterator = DataLoader(val_in, batch_size= 2)



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1,8,(5,5),stride=(2,3))
        #11x66x8
        self.conv2 = nn.Conv2d(8,16,(5,6),stride=(2,2))
        #4x31x16
        self.conv3 = nn.Conv2d(16,32,(2,5),stride=(2,2))
        #2x14x16
        self.linear1 = nn.Linear(in_features=896, out_features=5)
        self.activation_out = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation_out(self.conv1(x))
        x = self.activation_out(self.conv2(x))
        x = self.activation_out(self.conv3(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.activation_out(self.linear1(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(in_features=5, out_features=896)
        self.conv4 = nn.ConvTranspose2d(32,16,(2,5),stride=(2,2))
        self.conv5 = nn.ConvTranspose2d(16,8,(5,6),stride=(2,2))
        self.conv6 = nn.ConvTranspose2d(8,1,(5,5),stride=(2,3))
        self.activation_out = nn.LeakyReLU()
        self.activation_out1 = nn.Sigmoid()

    def forward(self, x):
        x = self.activation_out(self.linear2(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,32,2,14])
        x = self.activation_out(self.conv4(x))
        x = self.activation_out(self.conv5(x))
        x = self.activation_out(self.conv6(x))
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

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#encoder
encoder = Encoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)
    
optimizer = Adam(params=model.parameters(), lr=lr)

loss_crit  = nn.MSELoss()
train_losses = []
val_losses = []

def train():

    model.train()

    train_loss = 0.0

    for batch_ndx, x in enumerate(train_iterator):

        x = x.to(device)

        optimizer.zero_grad()

        predicted = model(x)

        loss = loss_crit(predicted, x)

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



        
        print('Epoch :',step, 'train_loss:',train_loss,':)')

train_losses = []
test_losses = []

for n_iter in range(N_EPOCHS):

    train_loss = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    test_loss /= len(test_iterator)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {n_iter}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')


plt.figure()
plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()



#save the models state dictionary for inference
model.eval()
torch.save(model.state_dict(),'Conv_AE_STATE_DICT.pt')

 