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
# plt.title('{} Layer'.format(i[g]))
plt.legend()
# tikzplotlib.save('/home/fusilly/ROM_using_Autoencoders/Bachelorarbeit/Figures/Layer_Sizes/{}.tex'.format(g))
plt.show()

