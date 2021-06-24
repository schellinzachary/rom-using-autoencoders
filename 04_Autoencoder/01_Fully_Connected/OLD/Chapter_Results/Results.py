'''
Plot results Linear 1.0
'''
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import tikzplotlib
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor




device = 'cpu'

qty = "hy" #["hy" or "rare"] Select to choose hydrodynamic or rarefied input data



class params:
    INPUT_DIM = 40
    H_SIZES = 40
    if qty == "hy":
        LATENT_DIM = 3
        num_mod = 3
    else:
        LATENT_DIM = 5
        num_mod = 5

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

def POD(c):
        u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
        S = np.diagflat(s)
        xx = u[:,:3]@S[:3,:3]@vh[:3,:]
        return xx, u






#INIT Model, Decoder and Encoder

encoder = net.Encoder()
decoder = net.Decoder()
model = net.Autoencoder(encoder, decoder).to(device)


#Load Model
if qty == "hy" :
    checkpoint = torch.load('/home/zachi/ROM_using_Autoencoders/04_Autoencoder/01_Fully_Connected/Parameterstudy/Hydro/04_Activations/Results/LeakyReLU_Tanh.pt')
else:
    checkpoint = torch.load('/home/zachi/ROM_using_Autoencoders/04_Autoencoder/01_Fully_Connected/Parameterstudy/Rare/04_Activations/Results/LeakyReLU_Tanh_test-4000.pt')

model.load_state_dict(checkpoint['model_state_dict'])
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']
N_EPOCHS = checkpoint['epoch']

#Load BGK FOM data
c_hy = np.load('/home/zachi/ROM_using_Autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
c_rare = np.load('/home/zachi/ROM_using_Autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy')
v = sio.loadmat('/home/zachi/ROM_using_Autoencoders/02_data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/zachi/ROM_using_Autoencoders/02_data_sod/sod25Kn0p00001/t.mat')
x = sio.loadmat('/home/zachi/ROM_using_Autoencoders/02_data_sod/sod25Kn0p00001/x.mat')
x = x['x']
x = x.squeeze()
t  = t['treport']
v = v['v']
t=t.squeeze()
t=t.T

if qty == "hy" :
    c = tensor(c_hy, dtype=torch.float)
else:
    c = tensor(c_rare, dtype=torch.float)

#---------------------------------------------------------------------------------------
#Inference with Autoencoder
predict,z = model(c)
c = c.detach().numpy()
predict = predict.detach().numpy()

#---------------------------------------------------------------------------------------
#POD
xx, u = POD(c)


#------------------------------------------------------------------------------------------
#Shape functions

def shapeback_code(z):
    c = np.empty((25,params.LATENT_DIM,200))
    n=0
    for i in range(25):
        for p in range(200):
          c[i,:,p] = z[p+n,:].detach().numpy()
        n += 200
    return(c) # shaping back the code

def shape_AE_code(g):
    
    c = np.empty((5000,3))
    for i in range(3):
        n = 0
        for t in range(25):
          print(n)
          c[n:n+200,i] = g[i][t,:]
          n += 200
    return(c)

def shapeback_field(predict):
    f = np.empty([25,40,200])
    n = 0
    for i in range(25):
        for j in range(200):
            f[i,:,j] = predict[j+n,:]
        n += 200
    return(f) # shaping back the field

#Conservation and Macroscopic quantities
#-----------------------------------------------------------------------------------------

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
    return(rho, rho_u, E) # calculate the macroscopic quantities of field

def conservativ(z):
    #g[:,0] = p, g[:,1] = rho, g[:,2] = u
    a = 1
    E = g[:,0] * a + g[:,1] * .5 * g[:,2]**2
    rho_u = g[:,1] * g[:,2] 
    rho = g[:,1]

    dt_E = diff(E)
    dt_rho_u = diff(rho_u)
    dt_rho = diff(rho)
    return(dt_E, dt_rho_u, dt_rho)

def plot_conservative_o_vs_p():
    predict_org_shape = shapeback_field(predict)
    original_org_shape = shapeback_field(c)

    macro_predict = macro(predict_org_shape,v) #macro_predict[0]=rho,macro_predict[1]=E,macro_predict[2]=rho_u
    macro_original = macro(original_org_shape,v) # " " "


    # dt_rho_o = np.diff(np.sum(rho_o,axis=1))/ np.mean(np.sum(rho_o,axis=(0,1)))
    # dt_rho_p = np.diff(np.sum(rho_p,axis=1))/ np.mean(np.sum(rho_p,axis=(0,1)))

    # dt_rho_u_p = np.diff(np.sum(rho_u_p,axis=1))/ np.mean(np.sum(rho_u_p,axis=(0,1)))
    # dt_rho_u_o = np.diff(np.sum(rho_u_o,axis=1))/ np.mean(np.sum(rho_u_o,axis=(0,1)))

    # dt_E_p = np.diff(np.sum(E_p,axis=1))/ np.mean(np.sum(E_p,axis=(0,1)))
    # dt_E_o = np.diff(np.sum(E_o,axis=1))/ np.mean(np.sum(E_o,axis=(0,1)))

    #derivative of conservatives in t mean
    # fig, ax = plt.subplots(1,3)
    # ax[0].plot(dt_rho_p,'-+''k',label=r'$y_p$')
    # ax[0].plot(dt_rho_o,'-v''k',label=r'$y_o$')
    # ax[0].set_xlabel(r'$t$')
    # ax[0].set_ylabel(r'$\hat{\rho}$')
    # ax[0].tick_params(axis='both')
    # ax[0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    # ax[0].legend()
    # ax[1].plot(dt_rho_u_p,'-+''k',label=r'$y_p$')
    # ax[1].plot(dt_rho_u_o,'-v''k',label=r'$y_o$')
    # ax[1].set_xlabel(r'$t$')
    # ax[1].set_ylabel(r'$\hat{\rho u}$')
    # ax[1].tick_params(axis='both')
    # ax[1].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    # ax[1].legend()
    # ax[2].plot(dt_E_p,'-+''k',label=r'$y_p$')
    # ax[2].plot(dt_E_o,'-v''k',label=r'$y_o$')
    # ax[2].set_xlabel(r'$t$')
    # ax[2].set_ylabel('$\hat{E}$')
    # ax[2].tick_params(axis='both')
    # ax[2].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    # ax[2].legend()
    # plt.show()

def plot_macro_2D():
    original_org_shape = shapeback_field(c)
    macro_original = macro(original_org_shape,v) 
        #macroscopic pcolor
    fig, ax = plt.subplots(3,1)
    for i in range(3):
        im = ax[i].imshow(macro_original[i],cmap='Greys',origin='lower')
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('t')
        ax[i].set_title('bla')
        colorbar = fig.colorbar(im, ax=ax[i])#,orientation='vertical')
        colorbar.set_ticks(np.linspace(np.min(macro_original[i]), np.max(macro_original[i]), 3))
        plt.tight_layout()
    ###### tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/Macrooriginal.tex')######
    plt.show()

def plot_macro_1D():
    original_org_shape = shapeback_field(c_hy)
    macro_original_hy = macro(original_org_shape,v)
    original_org_shape = shapeback_field(c_rare)
    macro_original_rare = macro(original_org_shape,v)
     
        #macroscopic pcolor
    fig, ax = plt.subplots(1,3)
    names = ['rho','rhu u','E']
    for i in range(3):
        ax[i].plot(x,macro_original_hy[i][-1],'k''-',label='Kn = 0.00001')
        ax[i].plot(x,macro_original_rare[i][-1],'k''--',label='Kn=0.01')
        ax[i].set_xlabel('x')
        ax[i].set_ylabel(names[i])
        ax[i].legend()
    plt.tight_layout()
    tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/01_Thesis/Figures/BGK/MacroFOMhyvsrare.tex')
    plt.show()
plot_macro_1D()
def plot_conservation_code_rho(z,t):
    g = shapeback_code(z)
    for i in range(params.LATENT_DIM):
        dt = np.gradient(g[:,i,:],axis=0) #dt
        dx = np.gradient(g[:,i,:],axis=1) #dx
        cons = np.sum(dt-dx,axis=0)
        plt.plot(cons)
        plt.plot()
        plt.xlabel('t')
        plt.ylabel('c{}'.format(i))
        plt.show()

#-----------------------------------------------------------------------------------------
#Calculate Characteristic

def characteritics(z,t):
    g = shapeback_code(z)
    #g = shapeback_field(z)
    #g ,a ,b = macro(g,t)
    u_x = np.sum(g,axis=2) #int dx
    s =[]
    for i in range(params.LATENT_DIM):
        f = np.gradient(u_x[:,i],axis=0)
        #s.append(f / (g[:,i,0] - g[:,i,-1]))
        s.append(f)
    # fig, ax = plt.subplots(1,params.LATENT_DIM)
    # for i in range(params.LATENT_DIM):
    #     ax[i].plot(s[i].T,'k''-*')
    #     ax[i].set_ylabel('u{}'.format(i))
    #     ax[i].set_xlabel('t')
        #ax[i].yaxis.set_ticks(np.linspace(np.min(s[i]), np.max(s[i]+1), params.LATENT_DIM))
    # # #tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/Characteristics.tex')
    # plt.imshow(g)
    # plt.plot(s[0]*np.linspace(0,25,25)+100,np.linspace(0,25,25))
    plt.show()
    return(s)

#Conservation of Code
#------------------------------------------------------------------------------------------

def plot_macro_vs_code(z,predict,x):
    g = shapeback_code(z)
    f = shapeback_field(predict)
    mac = macro(f,v)
    names = ['rho','rho u','E']
    timespots = [10,20]

    for i in timespots:
        fig , ax = plt.subplots(1,params.LATENT_DIM)
        for j in range(params.LATENT_DIM):
                ax[j].plot(x,g[i,j,:],'k''--',label='c{}'.format(j))
                ax[j].plot(x,mac[j][i,:],'k''-',label=names[j])
                #fig.suptitle('t={}'.format(t[i]))
                ax[j].set_xlabel('x')
                ax[j].set_ylabel('c_{},{}'.format(j,names[j]))
                ax[j].legend()
        ####tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/Hydro/MacroCode{}.tex'.format(i))  ###    
    plt.show()

#-----------------------------------------------------------------------------------------
#Interpolation results
def plot_interpolation(new_macro,old_macro):
    fig, ax = plt.subplots(1,3)
    names = ['rho','E','rho u']
    for i in range(3):
        ax[i].plot(new_macro[i][-1,:],'k''--',label='New Points')
        ax[i].plot(old_macro[i][-1,:],'k',label='Old Points')
        ax[i].set_ylabel(names[i])
        ax[i].set_xlabel('t')
        ax[i].legend(loc='upper right')
        #####tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/New_Points_Macro.tex')####
    plt.show()

#----------------------------------------------------------------------------------------
#Plot POD Modes and Autoencoder code
def plot_code(z,t,x):
    g = shapeback_code(z)
    #g = LA.norm(g)
    fig, ax = plt.subplots(params.LATENT_DIM,1)
    for i in range(params.LATENT_DIM):
        im = ax[i].imshow(g[:,i,:],extent=(0.0025,0.9975,0.0,0.12),label='g',cmap='Greys',origin='lower')
        ax[i].plot(s[i]*t+0.5,t)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('t')
        colorbar = fig.colorbar(im, ax=ax[i],orientation='vertical')
        colorbar.set_ticks(np.linspace(np.min(g[:,i,:]), np.max(g[:,i,:]), 3))
        ax[i].text(0.9, 0.07, 'c_{}'.format(i), bbox={'facecolor': 'white', 'pad': 2})
    #tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/Hydro/Code.tex') 
    plt.show()


    # fig, axs = plt.subplots(params.LATENT_DIM)

    # axs[0].plot(np.arange(5000),z[:,0].detach().numpy(),'k')
    # axs[0].set_ylabel('#')
    # axs[0].set_xlabel('x')

    # axs[1].plot(np.arange(5000),z[:,1].detach().numpy(),'k')
    # axs[1].set_ylabel('#')
    # axs[1].set_xlabel('x')

    # axs[2].plot(np.arange(5000),z[:,2].detach().numpy(),'k')
    # axs[2].set_ylabel('#')
    # axs[2].set_xlabel('x')

    # axs[3].plot(np.arange(5000),z[:,3].detach().numpy(),'k')
    # axs[3].set_ylabel('#')
    # axs[3].set_xlabel('x')

    # axs[4].plot(np.arange(5000),z[:,4].detach().numpy(),'k')
    # axs[4].set_ylabel('#')
    # axs[4].set_xlabel('x')

    plt.show()

def plot_pod_modes(v,u):
    fig, ax = plt.subplots(num_mod,1)
    for i in range(num_mod):
        ax[i].plot(v,u[:,i],'k')
        ax[i].set_xlabel('v')
        ax[i].set_ylabel('gamma{}'.format(i))
    ####tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/Hydro/PODModes.tex')#####
    plt.show()


#Bad Mistakes----------------------------------------------------------------------------
def bad_mistakes(c,predict):
    mistake_list = []
    for i in range(5000):
        mistake = np.sum(np.abs(c[i] - predict[i]))
        mistake_list.append((i,mistake))

    zip(mistake_list)

    plt.plot(c[900],'-o''m',label='$Original$')
    plt.plot(predict[900],'-v''k',label='$Prediction$')
    plt.xlabel('$Velocity$')
    plt.ylabel('$Probability$')
    plt.legend()
    plt.show()

    # np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_500_c.txt',c[500])
    # np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_500_p.txt',predict[500])
    # np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_Samples_1_1_lin.txt',mistake_list)
    #theta = np.linspace(0.0,2*np.pi,5000,endpoint=False)
    #width = (2*np.pi) / 5000
    ax = plt.subplot(111, polar=False)
    bars = ax.bar(range(len(mistake_list)),[val[1]for val in mistake_list],color='k',width=1)
    axr = ax.twiny()    
    axr.xaxis.set_major_locator(plt.FixedLocator(np.arange(0,25)))
    axr.set_xlim((0,25))
    ax.set_xlim((0,2499))
    ax.yaxis.grid(True)
    axr.xaxis.grid(True)
    ax.set_xlabel(r'$Samples$')
    axr.set_xlabel(r'$Timesteps$')
    ax.set_ylabel(r'$Absolute Error$')
    plt.show()


# -------------------------------------------------------------------------------------------
#Resulting reconstruction Error
ph_error = LA.norm((c - predict).flatten())/LA.norm(c.flatten())
print(ph_error)






