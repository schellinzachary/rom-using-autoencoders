import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor
import matplotlib.animation as animation
from scipy.interpolate import interp1d, Akima1DInterpolator, BarycentricInterpolator, PPoly, PchipInterpolator, KroghInterpolator
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':17})
rc('text', usetex=True)
#plt.rcParams['xtick.labelsize']=17
fonsize = 17

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




 
    checkpoint = torch.load('/home/fusilly/ROM_using_Autoencoders/Neural_Network/1_Lin_AE_Nets/Learning_Rate_Batch_Size/SD_kn_0p00001/AE_SD_5.pt')

    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']

    W = encoder.state_dict()['linear2.weight']

    #-------------------------------------------------------------------------------------------
    #Inference---------------------------------------------------------------------------------
    c = tensor(c, dtype=torch.float)

    predict,z = model(c)
    c = c.detach().numpy()
    predict = predict.detach().numpy()

    return predict, W, z

# load original data-----------------------------------------------------------------------
c = np.load('/home/fusilly/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
v = sio.loadmat('/home/fusilly/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/fusilly/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/t.mat')
t  = t['treport']
v = v['v']
t.squeeze()
t=t.T
#Predict the Data
prediction, W, z = net(c)

#Load Data
c_SVD=c.T
#SVD
def SVD(c):
	u, s, vh = np.linalg.svd(c_SVD,full_matrices=False) #s Singularvalues
	S = np.diagflat(s)
	xx = u[:,:3]@S[:3,:3]@vh[:3,:]
	return(xx)

def shapeback_field(predict):
    f = np.empty([25,40,200])
    n = 0
    for i in range(25):
        for j in range(200):
            f[i,:,j] = predict[j+n,:]
        n += 200
    return(f) # shaping back the field

def macro(f,v):
    dv = v[1]- v[0]
    rho = np.sum(f,axis = 1) * dv

    rho_u = f * v
    rho_u = np.sum(rho_u,axis = 1) * dv
    u = rho_u / rho

    E = f * ((v**2) * .5)
    E = np.sum(E, axis = 1)

    T = ((2* E) / (3 * rho)) - (u**2 / 3)
    p = rho * T
    return(rho, E, rho_u) # calculate the macroscopic quantities of field



xx  = SVD(c_SVD)
#Conservation of Prediction vs. Original
def plot_conservative_o_vs_p(predict, c, xx):
    predict_org_shape = shapeback_field(predict)
    original_org_shape = shapeback_field(c)
    SVD_predict_org_shape = shapeback_field(xx.T)

    rho_p, E_p, rho_u_p = macro(predict_org_shape,v)
    rho_psv, E_psv, rho_u_psv = macro(SVD_predict_org_shape,v)
    rho_o, E_o, rho_u_o = macro(original_org_shape,v)

    dt_rho_psv = np.gradient(np.sum(rho_psv,axis=1))/ np.mean(np.sum(rho_psv,axis=(0,1)))
    dt_rho_p = np.gradient(np.sum(rho_p,axis=1))/ np.mean(np.sum(rho_p,axis=(0,1)))
    dt_rho_o = np.gradient(np.sum(rho_o,axis=1))/ np.mean(np.sum(rho_o,axis=(0,1)))

    dt_rho_u_psv = np.gradient(np.sum(rho_u_psv,axis=1))/ np.mean(np.sum(rho_u_psv,axis=(0,1)))
    dt_rho_u_p = np.gradient(np.sum(rho_u_p,axis=1))/ np.mean(np.sum(rho_u_p,axis=(0,1)))
    dt_rho_u_o = np.gradient(np.sum(rho_u_o,axis=1))/ np.mean(np.sum(rho_u_o,axis=(0,1)))

    dt_E_psv = np.gradient(np.sum(E_psv,axis=1))/ np.mean(np.sum(E_psv,axis=(0,1)))
    dt_E_p = np.gradient(np.sum(E_p,axis=1))/ np.mean(np.sum(E_p,axis=(0,1)))
    dt_E_o = np.gradient(np.sum(E_o,axis=1))/ np.mean(np.sum(E_o,axis=(0,1)))


    fig, ax = plt.subplots(1,3)
    ax[0].plot(dt_rho_p,'-+''k',label=r'$y_p$')
    ax[0].plot(dt_rho_o,'-v''k',label=r'$y_o$')
    ax[0].plot(dt_rho_psv,'-o''k',label=r'$y_psv$')
    ax[0].set_xlabel(r'$t$',fontsize=fonsize)
    ax[0].set_ylabel(r'$\hat{\rho}$',fontsize=fonsize)
    ax[0].tick_params(axis='both', labelsize=fonsize)
    ax[0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    ax[0].legend()
    ax[1].plot(dt_rho_u_p,'-+''k',label=r'$y_p$')
    ax[1].plot(dt_rho_u_o,'-v''k',label=r'$y_o$')
    ax[1].plot(dt_rho_u_psv,'-o''k',label=r'$y_psv$')
    ax[1].set_xlabel(r'$t$',fontsize=fonsize)
    ax[1].set_ylabel(r'$\hat{\rho u}$',fontsize=fonsize)
    ax[1].tick_params(axis='both', labelsize=fonsize)
    ax[1].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    ax[1].legend()
    ax[2].plot(dt_E_p,'-+''k',label=r'$y_p$')
    ax[2].plot(dt_E_o,'-v''k',label=r'$y_o$')
    ax[2].plot(dt_E_psv,'-o''k',label=r'$y_psv$')
    ax[2].set_xlabel(r'$t$',fontsize=fonsize)
    ax[2].set_ylabel('$\hat{E}$',fontsize=fonsize)
    ax[2].tick_params(axis='both', labelsize=fonsize)
    ax[2].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    ax[2].legend()
    plt.show()
plot_conservative_o_vs_p(prediction, c, xx)

xx  = SVD(c_SVD)
print(xx.shape)