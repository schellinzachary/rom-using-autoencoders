'''
Parameterstudy_01_Layer_Size
'''

import numpy as np
import torch
import torch.nn as nn
import torch.tensor as tensor
import matplotlib.pyplot as plt
import scipy.io as sio
import tikzplotlib

device = 'cuda' if torch.cuda.is_available() else 'cpu'




class data():
    #load data
    f = np.load('/home/fusilly/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
    f = tensor(f, dtype=torch.float).to(device)

for g in range(5):
    #g=4
    class params():
        N_EPOCHS = 2000
        BATCH_SIZE = 16
        INPUT_DIM = 40
        H_SIZES = [[40,20,10,5],[40,20,10],[40,20],[40],[]]
        H_SIZE = H_SIZES[g]
        h_layers = len(H_SIZES[g])
        LATENT_DIM = 3
        lr = 1e-4
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            sizes = [y for x in [[params.INPUT_DIM], params.H_SIZE] for y in x]
            for l in range(params.h_layers):
                self.add_module('layer_' + str(l), torch.nn.Linear(in_features=sizes[l],out_features=sizes[l+1]))
                # if sizes[l] != 40:
                self.add_module('activ_' + str(l), nn.LeakyReLU())
            self.add_module('layer_c',nn.Linear(in_features=sizes[-1], out_features=params.LATENT_DIM))
            self.add_module('activ_c', nn.Tanh())
        def forward(self, x):
            for _, method in self.named_children():
                x = method(x)
            return x



    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            sizes = [y for x in [[params.INPUT_DIM], params.H_SIZE] for y in x]
            sizes.reverse()
            self.add_module('layer_c',nn.Linear(in_features=params.LATENT_DIM, out_features=sizes[0]))
            self.add_module('activ_c', nn.LeakyReLU())
            for l in range(params.h_layers):
                self.add_module('layer_' + str(l), nn.Linear(in_features=sizes[l],out_features=sizes[l+1]))
                if sizes[l] != 40:
                    self.add_module('activ_' + str(l), nn.LeakyReLU())
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
            return predicted

    #encoder
    encoder = Encoder()

    #decoder
    decoder = Decoder()

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    checkpoint = torch.load('Results/LS_{}.pt'.format(g))

    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']

    rec = model(data.f)

    l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
    print(l2_error)
    print(len(test_losses))
    print(N_EPOCHS)

    plt.semilogy(train_losses,'k''--',label='Train')
    plt.semilogy(test_losses,'k''-',label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    ax = plt.gca()
    i = [10,8,6,4,2]
    plt.title('{} Layer'.format(i[g]))
    plt.legend()
    tikzplotlib.save('/home/fusilly/ROM_using_Autoencoders/Bachelorarbeit/Figures/Layer_Sizes/{}.tex'.format(g))
    plt.show()

