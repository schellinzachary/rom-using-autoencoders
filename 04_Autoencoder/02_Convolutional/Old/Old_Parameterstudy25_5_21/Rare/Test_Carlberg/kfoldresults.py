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
class data():
    #load data
    f = np.load('/home/zachi/ROM_using_Autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p01_4D_unshuffled.npy')
    f = tensor(f, dtype=torch.float).to(device)


for i in [0,1,2,3,4]:



    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.convE1 = nn.Conv2d(1,4,5,stride=2,padding=(1,1))
            self.convE2 = nn.Conv2d(4,8,5,stride=2,padding=(1,1))
            self.convE3 = nn.Conv2d(8,16,5,stride=2,padding=(1,1))
            self.convE4 = nn.Conv2d(16,32,5,stride=2,padding=(2,1))
            self.linearE1 = nn.Linear(in_features=352,out_features=5)
            self.add_module('act',nn.SiLU())
            # self.add_module('act_c',act_c_list[i])

        def forward(self, x):
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            x = self.act(self.convE3(x))
            x = self.act(self.convE4(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            # x = self.act_c(self.linearE1(x))
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
            # self.add_module('act_c',act_c_list[i])

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
            z= self.enc(x)
            x = self.dec(z)
            return x, z
    #encoder
    encoder = Encoder()

    #decoder
    decoder = Decoder()

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    # checkpoint = torch.load('Results/{}.pt'.format((i,act_c_list[i])),map_location='cpu')
    checkpoint = torch.load('Results/fold{}.pt'.format(i),map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']

    rec, z = model(data.f)
    # fig,ax = plt.subplots(3,1)
    # ax[0].imshow(rec[39,0,:,:].detach().numpy())
    # ax[1].imshow(data.f[39,0,:,:].detach().numpy())

    # fige , ax = plt.subplots(5,1)
    # for i in range(5):
    #     ax[i].plot(z[:,i].detach().numpy())
    # plt.show()

    l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
    # print('{}:'.format((i,act_c_list[i])),l2_error)
    print('{}:'.format(i),l2_error)

    a = ['-o','-x','-v','-<']      
    plt.semilogy(train_losses,'k')#'-{}'.format(a[act_list.index(i)]),label='Design {} Train'.format(i),markevery=400)
    plt.semilogy(test_losses,'k')#'{}'.format(a[act_list.index(i)]),label='Design {} Validate'.format(i),markevery=400)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    #ax = ax[i].gca()
    plt.title('Train & validation for channel designs with 4 layers')
    plt.legend()
    #tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Parameterstudy/Convolutional/Activations/L4R.tex')
    plt.show()

