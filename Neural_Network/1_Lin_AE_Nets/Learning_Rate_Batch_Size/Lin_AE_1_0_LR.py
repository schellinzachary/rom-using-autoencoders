'''
Linear Autoencoder v1.0
'''


import sys
import numpy as np
from numpy import linalg as LA
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io as sio
from random import randint


def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.is_available())

for i in range(6):

    a = (64, 32, 16, 8, 4, 2)

    N_EPOCHS = 1000
    BATCH_SIZE = a[i] 
    INPUT_DIM = 40
    HIDDEN_DIM = 20
    LATENT_DIM = 3
    lr = 1e-4

    #print('learning rate :',lr,'batch size :', BATCH_SIZE)



    #load data
    f = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_2D.npy')

    f = tensor(f, dtype=torch.float).to(device)

    train_in = f[0:4000]
    val_in = f[4000:5000]

    train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
    test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))

    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, lat_dim):
            super(Encoder, self).__init__()
            self.linear1 = nn.Linear(in_features=input_dim, 
                                        out_features=hidden_dim)
            self.linear2 = nn.Linear(in_features=hidden_dim, 
                                        out_features=lat_dim)
            self.act = nn.LeakyReLU()
            self.actc= nn.Tanh()
        def forward(self, x):
            x = self.act(self.linear1(x))
            x = self.actc(self.linear2(x))
            return x


    class Decoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, lat_dim):
            super(Decoder, self).__init__()
            self.linear3 = nn.Linear(in_features=lat_dim, 
                                    out_features=hidden_dim)
            self.linear4 = nn.Linear(in_features=hidden_dim, 
                                    out_features=input_dim)
            self.act = nn.LeakyReLU()
            self.actc = nn.Tanh()

        def forward(self,x):
            x = self.actc(self.linear3(x))
            x = self.act(self.linear4(x))
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


    #encoder
    encoder = Encoder(INPUT_DIM,HIDDEN_DIM, LATENT_DIM)

    #decoder
    decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    #model.load_state_dict(torch.load('Lin_AE_STATE_DICT_1_0_L5_sigmoid.pt'))
       
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

            return test_loss

    test_losses = []
    val_losses = []

    for epoch in range(N_EPOCHS):

        train_loss = train()
        test_loss  = test()

        #save and print the loss
        train_loss /= len(train_iterator)
        test_loss /= len(test_iterator)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        progressBar(epoch,N_EPOCHS)

        #print(f'Epoch {n_iter}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}')


        # if n_iter % 300 == 0:

        #      i = randint(0,999)
        #      x = val_in[i].to(device)

        #      predicted = model(x)
        #      x = x.to('cpu')
        #      predicted = predicted.to('cpu')
        #      data = x.detach().numpy()
        #      predict = predicted.detach().numpy()
            
        #      plt.plot(x, label='Original')
        #      plt.plot(predict, label='Predicted')
        #      plt.legend()
        #      plt.show()

    # plt.figure()
    # plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
    # plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
    # plt.legend(loc='upper right')
    # plt.xlabel('trainstep')
    # plt.ylabel('loss')
    # plt.show()

    #Inference

    rec = model(f)


    ph_error = torch.norm((f - rec).flatten())/torch.norm(f.flatten())
    print('Batch_Size:', BATCH_SIZE)
    print('ph_error:', ph_error)



    #save the models state dictionary for inference
    torch.save({
        'epoch': epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_losses':train_losses,
        'test_losses': test_losses,
        'ph_error' : ph_error,
        'batch_size' : BATCH_SIZE,
        'learning-rate' : lr
        },f'SD_kn_0p01/AE_SD_%s.pt'%i)

