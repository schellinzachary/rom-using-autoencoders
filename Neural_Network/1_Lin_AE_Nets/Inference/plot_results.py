'''
Plot results Concolutional
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor

INPUT_DIM = 25*200
HIDDEN_DIM = 500
LATENT_DIM = 5


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1,4,(5,5),stride=(2,3))
        #11x66x8
        self.conv2 = nn.Conv2d(4,8,(5,6),stride=(2,2))
        #4x31x16
        self.conv3 = nn.Conv2d(8,16,(2,5),stride=(2,2))
        #2x14x16
        self.linear1 = nn.Linear(in_features=448, out_features=5)
        self.activation_out = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation_out(self.conv1(x))
        x = self.activation_out(self.conv2(x))
        x = self.activation_out(self.conv3(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.activation_out(self.linear1(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(in_features=5, out_features=448)
        self.conv4 = nn.ConvTranspose2d(16,8,(2,5),stride=(2,2))
        self.conv5 = nn.ConvTranspose2d(8,4,(5,6),stride=(2,2))
        self.conv6 = nn.ConvTranspose2d(4,1,(5,5),stride=(2,3))
        self.activation_out = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation_out(self.linear2(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,16,2,14])
        x = self.activation_out(self.conv4(x))
        x = self.activation_out(self.conv5(x))
        x = self.activation_out(self.conv6(x))
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

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#encoder
encoder = Encoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder)


torch.load('Conv_AE_STATE_DICT.pt')


# load original data
data = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/f.mat')
data = data['f']
data = np.array(data, order='C')

data = np.reshape(data,(40,25,200))
data = np.expand_dims(data, axis=1)
data = tensor(data, dtype=torch.float)


predict = model(data)
predict = predict.squeeze()
data = data.squeeze()
data = data.detach().numpy()
predict = predict.detach().numpy()

data = np.reshape(data, (25,40,200))
predict = np.reshape(predict, (25,40,200))



rho_predict = np.zeros([25,200])
rho_samples = np.zeros([25,200])
for i in range(25):
    for k in range(200):
        rho_samples[i,k] = np.sum(data[i,:,k]) * 0.5128
        rho_predict[i,k] = np.sum(predict[i,:,k]) * 0.5128



plt.ion()
plt.figure()
for i in range(25):   
    plt.plot(rho_predict[i,:],label='Predicted')
    plt.plot(rho_samples[i,:],label='Original')
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('A * sin(g*x + t)')
    plt.draw()
    plt.pause(0.3)
    plt.clf()

