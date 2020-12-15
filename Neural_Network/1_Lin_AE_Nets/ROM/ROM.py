'''
Plot results Linear 1.0
'''
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor
import matplotlib.animation as animation


def net(c):

    INPUT_DIM = 40
    HIDDEN_DIM = 20
    LATENT_DIM = 3


    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, lat_dim):
            super(Encoder, self).__init__()

            self.linear1 = nn.Linear(in_features=input_dim, 
                                        out_features=hidden_dim)
            self.linear2 = nn.Linear(in_features=hidden_dim, 
                                        out_features=lat_dim)
            self.act = nn.LeakyReLU()
            self.actc = nn.Tanh()

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
            self.act= nn.LeakyReLU()
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
            return predicted, z




    #encoder
    encoder = Encoder(INPUT_DIM,HIDDEN_DIM, LATENT_DIM)

    #decoder
    decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    #Autoencoder
    model = Autoencoder(encoder, decoder)




 
    checkpoint = torch.load('/home/fusilly/ROM_using_Autoencoders/Neural_Network/1_Lin_AE_Nets/Parameterstudy/Learning_Rate_Batch_Size/SD_kn_0p00001/AE_SD_5.pt')

    model.load_state_dict(checkpoint['model_state_dict'])


    c = tensor(c, dtype=torch.float)
    predict,z = model(c)
    c = c.detach().numpy()
    predict = predict.detach().numpy()


    return predict, z

# load original data-----------------------------------------------------------------------
c = np.load('/home/fusilly/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
v = sio.loadmat('/home/fusilly/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/fusilly/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/t.mat')
t  = t['treport']
v = v['v']
t.squeeze()
t=t.T
#Inference-----------------------------------------------------------------------------------
predict, z = net(c)

def characteritics(z,t):
    g = shapeback_code(z)
    u_x = np.sum(g,axis=2)
    f1 = np.gradient(u_x[:,0],0.005)
    f2 = np.gradient(u_x[:,1],0.005)
    f3 = np.gradient(u_x[:,2],0.005)
    s1 = f1  / (g[:,0,0] - g[:,0,-1])
    s2 = f2 / (g[:,1,0] - g[:,1,-1])
    s3 = f3 / (g[:,2,0] - g[:,2,-1])
    plt.plot(s1)
    plt.plot(s2)
    plt.plot(s3)
    plt.show()

def shapeback_code(z): # shaping back the code
    c = np.empty((25,3,200))
    n=0
    for i in range(25):
        for p in range(200):
          c[i,:,p] = z[p+n,:].detach().numpy()
        n += 200
    return c

c = shapeback_code(z)

np.save('abc',c)
