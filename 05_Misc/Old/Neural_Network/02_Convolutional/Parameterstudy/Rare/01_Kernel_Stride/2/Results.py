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
    f = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_4D_unshuffled.npy')
    f = tensor(f, dtype=torch.float).to(device)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.m = nn.ZeroPad2d((0,0,1,1))
        self.convE1 = nn.Conv2d(1,8,(6,10),stride=(3,10))
        self.convE2 = nn.Conv2d(8,16,(4,10),stride=(4,10))
        self.linearE1 = nn.Linear(in_features=64,out_features=5)
        self.act = nn.Tanh()
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
        self.linearD1 = nn.Linear(in_features=5, out_features=64)
        self.convD1 = nn.ConvTranspose2d(16,8,(4,10),stride=(4,10))
        self.convD2 = nn.ConvTranspose2d(8,1,(4,10),stride=(3,10))
        self.act = nn.Tanh()
        #self.act_c = nn.Tanh()

    def forward(self, x):
        x = self.linearD1(x)
        #x = self.act_c(self.linearD1(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,2,2])
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

checkpoint = torch.load('Results/1.pt')

model.load_state_dict(checkpoint['model_state_dict'])
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']
N_EPOCHS = checkpoint['epoch']

rec = model(data.f)

l2_error = torch.norm((data.f - rec).flatten())/torch.norm(data.f.flatten())
print(l2_error)


plt.semilogy(train_losses,'k''--',label='Train')
plt.semilogy(test_losses,'k''-',label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
ax = plt.gca()
plt.title('2 Layers')
plt.legend()
tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Parameterstudy/Convolutional/Layers/2Layer.tex')
plt.show()
