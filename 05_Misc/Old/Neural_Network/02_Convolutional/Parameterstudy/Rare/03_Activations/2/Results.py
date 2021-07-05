
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

act_list = [nn.LeakyReLU(),nn.SiLU(),nn.ELU()]#,nn.ReLU()]
# act_list = [nn.LeakyReLU(),nn.SiLU(),nn.SiLU(),nn.SiLU(),nn.ELU()]
# act_c_list = [nn.Tanh(),nn.LeakyReLU(),nn.Tanh(),nn.ELU(),nn.SiLU()]

for i in act_list:
    class data():
        #load data
        f = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_4D_unshuffled.npy')
        f = tensor(f, dtype=torch.float).to(device)


    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.m = nn.ZeroPad2d((0,0,1,1))
            self.convE1 = nn.Conv2d(1,32,(6,10),stride=(3,10))
            self.convE2 = nn.Conv2d(32,64,(4,10),stride=(4,10))
            self.linearE1 = nn.Linear(in_features=256,out_features=5)
            self.add_module('act',i)
            # self.add_module('act_c',act_c_list[i])

        def forward(self, x):
            x = self.m(x)
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            # x = self.act_c(self.linearE1(x))
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=5, out_features=256)
            self.convD1 = nn.ConvTranspose2d(64,32,(4,10),stride=(4,10))
            self.convD2 = nn.ConvTranspose2d(32,1,(4,10),stride=(3,10))
            self.add_module('act',i)
            # self.add_module('act_c',act_c_list[i])

        def forward(self, x):
            x = self.linearD1(x)
            # x = self.act_c(self.linearD1(x))
            dim = x.shape[0]
            x = torch.reshape(x,[dim,64,2,2])
            x = self.act(self.convD1(x))
            x = self.convD2(x)
            return x



    class Autoencoder(nn.Module):
        def __init__(self, enc, dec):
            super(Autoencoder, self).__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            z = self.encoder(x)
            x = self.decoder(z)
            return x, z


    #encoder
    encoder = Encoder()

    #decoder
    decoder = Decoder()

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    # checkpoint = torch.load('Results/{}.pt'.format((i,act_c_list[i])),map_location='cpu')
    checkpoint = torch.load('Results/nll32{}.pt'.format(i),map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']

    rec, z = model(data.f)

    # fig,ax = plt.subplots(5,1)
    # for i in range(5):
    #     ax[i].plot(z[:,i].detach().numpy())
    #     ax[i].set_title('{}'.format(i))
    # plt.show()

    l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
    # print('{}:'.format((i,act_c_list[i])),l2_error)
    print('{}:'.format(i),l2_error)


    a = ['-o','-x','-v','-<']      
    plt.semilogy(train_losses,'k''-{}'.format(a[act_list.index(i)]),label='Design {} Train'.format(i),markevery=400)
    plt.semilogy(test_losses,'k''{}'.format(a[act_list.index(i)]),label='Design {} Validate'.format(i),markevery=400)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    #ax = ax[i].gca()
    plt.title('Train & validation for channel designs with 2 layers')
    plt.legend()
    # tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Parameterstudy/Convolutional/Activations/L2R.tex')
plt.show()

