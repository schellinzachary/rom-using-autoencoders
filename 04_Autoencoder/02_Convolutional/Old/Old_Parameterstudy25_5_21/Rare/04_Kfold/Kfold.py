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
from sklearn.model_selection import KFold



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
f = np.load('/home/zachi/ROM_using_Autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p01_4D.npy')
f = tensor(f, dtype=torch.float).to(device)

kf = KFold(n_splits=5)
# for train, test in kf.split(f):
#    print("%s %s" % (train, test))
i = 0
for train, test in kf.split(f):
    print(i)
    train_in = f[train]
    val_in = f[test]

    train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
    test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))





    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.convE1 = nn.Conv2d(1,4,5,stride=2,padding=(1,1))
            self.convE2 = nn.Conv2d(4,8,5,stride=2,padding=(1,1))
            self.convE3 = nn.Conv2d(8,16,5,stride=2,padding=(1,1))
            self.convE4 = nn.Conv2d(16,32,5,stride=2,padding=(2,1))
            self.linearE1 = nn.Linear(in_features=352,out_features=5)
            self.add_module('act',nn.SiLU())


        def forward(self, x):
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            x = self.act(self.convE3(x))
            x = self.act(self.convE4(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
   
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=5, out_features=352)
            self.convD1 = nn.ConvTranspose2d(32,16,5,stride=2,padding=(2,0)) 
            self.convD2 = nn.ConvTranspose2d(16,8,5,stride=2,padding=(0,2)) 
            self.convD3 = nn.ConvTranspose2d(8,4,5,stride=2,padding=(1,1)) 
            self.convD4 = nn.ConvTranspose2d(4,1,(5,4),stride=2,)
            self.add_module('act',nn.SiLU())


        def forward(self, x):
            x = self.linearD1(x)
            # x = self.act_c(self.linearD1(x))
            dim = x.shape[0]
            x = torch.reshape(x,[dim,32,1,11])
            x = self.act(self.convD1(x))
            x = self.act(self.convD2(x))
            x = self.act(self.convD3(x))
            x = self.act(self.convD4(x))
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


    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600], gamma=0.1)


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

                scheduler.step()

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
        },'Results/fold{}.pt'.format(i))
    i +=1


