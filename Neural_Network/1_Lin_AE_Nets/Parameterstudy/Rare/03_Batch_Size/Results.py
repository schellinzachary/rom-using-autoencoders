import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
import matplotlib.pyplot as plt
import tikzplotlib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'

for g in range(7):
    class params():
        INPUT_DIM = 40
        H_SIZES = 40
        LATENT_DIM = 5
    class data():
        #load data
        f = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy')
        f = tensor(f, dtype=torch.float).to(device)

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.add_module('layer_1', torch.nn.Linear(in_features=params.INPUT_DIM,out_features=params.H_SIZES))
            self.add_module('activ_1', nn.LeakyReLU())
            self.add_module('layer_c',nn.Linear(in_features=params.H_SIZES, out_features=params.LATENT_DIM))
            self.add_module('activ_c', nn.Tanh())
        def forward(self, x):
            for _, method in self.named_children():
                x = method(x)
            return x



    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.add_module('layer_c',nn.Linear(in_features=params.LATENT_DIM, out_features=params.H_SIZES))
            self.add_module('activ_c', nn.LeakyReLU())
            self.add_module('layer_4', nn.Linear(in_features=params.H_SIZES,out_features=params.INPUT_DIM))
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

    model = Autoencoder(encoder, decoder).to(device)
    i = [2,4,6,8,10,16,32]
    checkpoint = torch.load('Results/{}.pt'.format(i[g]), map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']
    batch_size = checkpoint['batch_size']
    print(batch_size)

    rec = model(data.f)

    l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
    print(l2_error)

    plt.semilogy(train_losses,'k''--',label='Train')
    plt.semilogy(test_losses,'k''-',label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    ax = plt.gca()
    
    plt.title('Mini-Batch Size {}'.format(i[g]))
    plt.legend()
    tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Parameterstudy/Batch_Size/Rare/{}.tex'.format(i[g]))
    plt.show()