'''
Plot results Linear 1.0
'''
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor



device = 'cpu'

qty = "hy" #["hy" or "rare"]


class params:
    INPUT_DIM = 40
    H_SIZES = 40
    if qty == "hy":
        LATENT_DIM = 3
    else:
        LATENT_DIM = 5

class net:

    class Encoder(nn.Module):
        def __init__(self):
            super(net.Encoder, self).__init__()
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
            super(net.Decoder, self).__init__()
            self.add_module('layer_c',nn.Linear(in_features=params.LATENT_DIM, out_features=params.H_SIZES))
            self.add_module('activ_c', nn.LeakyReLU())
            self.add_module('layer_4', nn.Linear(in_features=params.H_SIZES,out_features=params.INPUT_DIM))
        def forward(self, x):
            for _, method in self.named_children():
                x = method(x)
            return x

            # def forward(self,x):
            #     x = self.actc(self.linear3(x))
            #     x = self.act(self.linear4(x))
            #     return x
            # def predict(self, dat_obj):
            #     y_predict = self.forward(x_predict)
            #     return(y_predict)
        


    class Autoencoder(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, x):
            z = self.enc(x)
            predicted = self.dec(z)
            return predicted, z



#INIT Model, Decoder and Encoder
#encoder
encoder = net.Encoder()

#decoder
decoder = net.Decoder()

#Autoencoder
model = net.Autoencoder(encoder, decoder).to(device)


#Load Model params
if qty == "hy" :
    checkpoint = torch.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/01_Fully_Connected/Parameterstudy/Hydro/04_Activations/Results/LeakyReLU_Tanh.pt')
else:
    checkpoint = torch.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/01_Fully_Connected/Parameterstudy/Rare/04_Activations/Results/LeakyReLU_Tanh_test-4000.pt')

model.load_state_dict(checkpoint['model_state_dict'])
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']
N_EPOCHS = checkpoint['epoch']
# load original data-----------------------------------------------------------------------
c_hy = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
c_rare = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy')
v = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/t.mat')
x = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/x.mat')
x = x['x']
t  = t['treport']
v = v['v']
t=t.squeeze()
t=t.T

#Inference---------------------------------------------------------------------------------
if qty == "hy" :
    c = tensor(c_hy, dtype=torch.float)
else:
    c = tensor(c_rare, dtype=torch.float)


predict,z = model(c)
c = c.detach().numpy()
predict = predict.detach().numpy()

def shapeback_code(z):
    c = np.empty((25,params.LATENT_DIM,200))
    n=0
    for i in range(25):
        for p in range(200):
          c[i,:,p] = z[p+n,:].detach().numpy()
        n += 200
    return(c) # shaping back the code

def shape_AE_code(g):
    c = np.empty((5000,params.LATENT_DIM))
    for i in range(params.LATENT_DIM):
        n = 0
        for t in range(25):
          c[n:n+200,i] = g[t,i,:]
          n += 200
    return(c)


def characteritics(g):
    u_x = np.sum(g,axis=2) #int dx
    s =[]
    for i in range(params.LATENT_DIM):
        f = np.gradient(u_x[:,i],axis=0)
        s.append(f.mean())

    return(s)

def int_code(s,g):
    t_new = np.linspace(0,0.12,13)
    g_new = np.empty((25,params.LATENT_DIM,200))
    for i in range(params.LATENT_DIM):
        for j in range(25):
            g_new[j,i,:] = (g[2,i,:]+s[i]*t[j])
    return(g_new)

#Interpolate

g = shapeback_code(z)
s = characteritics(g)
g_new = int_code(s,g)
g_new = shape_AE_code(g_new)
g_new = tensor(g_new,dtype=torch.float).to(device)
new_points = decoder(g_new)
plt.plot(new_points[:,5].detach().numpy())
plt.show()






