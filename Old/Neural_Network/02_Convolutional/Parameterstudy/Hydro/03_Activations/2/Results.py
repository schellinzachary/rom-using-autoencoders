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

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

# act_list = [nn.LeakyReLU(),nn.SiLU(),nn.ELU(),nn.ReLU()]
act_list = [nn.LeakyReLU(),nn.SiLU(),nn.ELU(),nn.SiLU()]
act_c_list = [nn.Tanh(),nn.ELU(),nn.Tanh(),nn.Tanh()]

for i in range(4):
    class data():
        #load data
        f = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_4D_unshuffled.npy')
        f = tensor(f, dtype=torch.float).to(device)


    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.m = nn.ZeroPad2d((0,0,1,1))
            self.convE1 = nn.Conv2d(1,32,(6,10),stride=(3,10))
            self.convE2 = nn.Conv2d(32,64,(4,10),stride=(4,10))
            self.linearE1 = nn.Linear(in_features=256,out_features=3)
            self.add_module('act',act_list[i])
            #self.act_c = nn.Tanh()

        def forward(self, x):
            x = self.m(x)
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            #x = self.act_c(self.linearE1(x))
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=3, out_features=256)
            self.convD1 = nn.ConvTranspose2d(64,32,(4,10),stride=(4,10))
            self.convD2 = nn.ConvTranspose2d(32,1,(4,10),stride=(3,10))
            self.add_module('act',act_list[i])
            #self.act_c = nn.Tanh()

        def forward(self, x):
            x = self.linearD1(x)
            #x = self.act_c(self.linearD1(x))
            dim = x.shape[0]
            x = torch.reshape(x,[dim,64,2,2])
            x = self.act(self.convD1(x))
            x = self.act(self.convD2(x))
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

    checkpoint = torch.load('Results/{}.pt'.format((act_list[i],act_c_list[i])),map_location='cpu')
    # checkpoint = torch.load('Results/{}.pt'.format(act_list[i]),map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']

    rec = model(data.f)
    fig,ax = plt.subplots(3,1)
    ax[0].imshow(rec[39,0,:,:].detach().numpy())
    ax[1].imshow(data.f[39,0,:,:].detach().numpy())


    l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
    print('{}:'.format((act_list[i],act_c_list[i])),l2_error)
    # print('{}:'.format(act_list[i]),l2_error)

    plt.figure(i)
    plt.semilogy(train_losses,'k''--',label='Train')
    plt.semilogy(test_losses,'k''-',label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    ax = plt.gca()
    plt.title('{}'.format((act_list[i],act_c_list[i])))
    # plt.title('{}'.format(act_list[i]))
    plt.legend()
    # tikzplotlib.save('/home/fusilly/ROM_using_Autoencoders/Bachelorarbeit/Figures/Layer_Sizes/{}.tex'.format(g))
plt.show()

