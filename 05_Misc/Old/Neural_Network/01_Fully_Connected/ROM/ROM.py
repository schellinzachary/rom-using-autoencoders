# DMD Interpolation



import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor


device = 'cpu'

qty = "rare" #["hy" or "rare"]


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


#Load Net and Data
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

#Inference-----------------------------------------------------------------------------------
predict, z = model(c)

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

def DMD(X,Xprime,r):
    U,Sigma,VT = np.linalg.svd(X,full_matrices=0) 
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]
    Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T).T).T
    Lambda, W = np.linalg.eig(Atilde) 
    Lambda = np.diag(Lambda)
    Phi = Xprime @ np.linalg.solve(Sigmar.T,VTr).T @ W 
    alpha1 = Sigmar @ VTr[:,0]
    b = np.linalg.solve(W @ Lambda,alpha1)
    return Phi, Lambda, b

z = shapeback_code(z)

it_interpolated = 8
it = it_interpolated*2
ivar=4
Ndat = z.shape
X=np.reshape(z, [25,-1])
print(X.shape)
X = X.transpose()
X_half=X[:,::2]
Phi, Lambda, b = DMD(X_half[:,:-1],X_half[:,1:],-1)

Xnew=Phi@np.exp(np.log(Lambda)*it_interpolated)@b

unew = np.reshape(Xnew,Ndat[1:])

plt.title("interpolation results for: %s" % qty)
plt.plot(z[it-1,ivar,:],'--', label = r"$t_{i-1}$")
plt.plot(z[it+1,ivar,:],'--',  label = r"$t_{i+1}$")
plt.plot(z[it,ivar,:],'-o', label = r"$t_{i}$ (truth)")
plt.plot(unew[ivar,:],'k+', label=r"$t_i$ (predicted)")
plt.legend()
plt.ylabel(r"$u_%d$"%ivar)
plt.xlabel(r"$x$")
plt.xlim([50,160])
plt.show()