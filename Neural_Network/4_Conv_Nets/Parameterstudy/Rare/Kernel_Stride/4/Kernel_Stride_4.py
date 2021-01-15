'''
Convolutional Autoencoder v1.0
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys



def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()



#torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

N_EPOCHS = 6000
BATCH_SIZE = 4
lr = 1e-4


#load data
f = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_4D.npy')
f = tensor(f, dtype=torch.float).to(device)

train_in = f[0:32]
val_in = f[32:40]

train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convE1 = nn.Conv2d(1,8,(3,2),stride=2)
        self.convE2 = nn.Conv2d(8,16,(2,2),stride=2)
        self.convE3 = nn.Conv2d(16,32,(2,2),stride=2)
        self.convE4 = nn.Conv2d(32,64,(3,3),stride=(1,2))
        self.linearE1 = nn.Linear(in_features=768,out_features=5)
        self.act = nn.Tanh()
        #self.act_c = nn.Tanh()

    def forward(self, x):
        x = self.act(self.convE1(x))
        x = self.act(self.convE2(x))
        x = self.act(self.convE3(x))
        x = self.act(self.convE4(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        #x = self.act_c(self.linearE1(x))
        x = self.linearE1(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linearD1 = nn.Linear(in_features=5, out_features=768)
        self.convD1 = nn.ConvTranspose2d(64,32,(3,3),stride=(1,2))
        self.convD2 = nn.ConvTranspose2d(32,16,(2,2),stride=2)
        self.convD3 = nn.ConvTranspose2d(16,8,(2,2),stride=2)
        self.convD4 = nn.ConvTranspose2d(8,1,(3,2),stride=2)
        self.act = nn.Tanh()
        #self.act_c = nn.Tanh()

    def forward(self, x):
        x = self.linearD1(x)
        #x = self.act_c(self.linearD1(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,64,1,12])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        x = self.act(self.convD3(x))
        x = self.act(self.convD4(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#encoder
encoder = Encoder()

#decoder
decoder = Decoder()

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

optimizer = Adam(params=model.parameters(), lr=lr)


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

            #scheduler.step(loss)

        return test_loss


train_losses = []
test_losses = []




for epoch in range(N_EPOCHS):
    train_loss = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    progressBar(epoch,N_EPOCHS)
#save the models state dictionary for inference
torch.save({
    'epoch': epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses':train_losses,
    'test_losses': test_losses,
    'model': model
    },'Results/1.pt')


