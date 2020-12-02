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
from scipy.interpolate import interp1d, Akima1DInterpolator, BarycentricInterpolator, PPoly, PchipInterpolator, KroghInterpolator
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':15})

# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
plt.rcParams['xtick.labelsize']=15

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




 
    checkpoint = torch.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/1_Lin_AE_Nets/Learning_Rate_Batch_Size/SD_kn_0p00001/AE_SD_5.pt')

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
c = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
v = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/t.mat')
t  = t['treport']
v = v['v']
t.squeeze()
t=t.T
#Inference-----------------------------------------------------------------------------------
predict, W, z = net(c)

#------------------------------------------------------------------------------------------
#Conservation Properties

def shapeback_code(z):
    c = np.empty((25,200,3))
    n=0
    for i in range(25):
        for p in range(200):
          c[i,p,:] = z[p+n,:].detach().numpy()
        n += 200
    return(c) # shaping back the code

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

def diff(r):
        dt = torch.empty(25)
        for i in range(25):
            dt[i]= r[i+1] - r[i] 
        #dt = dt / torch.mean(r)
        return(dt)

def energy(g):
    a=2
    E = np.sum(g[:,:,2],axis=1) * a + np.sum(g[:,:,1],axis=1) * .5 * np.sum(g[:,:,0]**2,axis=1)

    return(E) # cal # calculate Energy of code

predict_org_shape = shapeback_field(predict)
original_org_shape = shapeback_field(c)

g = shapeback_code(z)

rho_p, E_p, rho_u_p = macro(predict_org_shape,v)
rho_o, E_o, rho_u_o = macro(original_org_shape,v)

d_dt_p = np.diff(np.sum(rho_u_p,axis=1))/ np.mean(np.sum(rho_u_p,axis=(0,1)))
d_dt_o = np.diff(np.sum(rho_u_o,axis=1))/ np.mean(np.sum(rho_u_o,axis=(0,1)))

dt_E_code, dt_rho_u_code, dt_rho_code = conservativ(z)

E_code = energy(g)

plt.plot(dt_rho_code,'-+''k')
#plt.plot(d_dt_o,'-v''k')
plt.xlabel('t',fontsize=25)
plt.ylabel('(dp/dt)',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
plt.show()



print('ggg',np.sum(np.abs(g))-np.sum(np.abs(z.detach().numpy())))
plt.pcolor(g[:,:,2])
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()


# plot code-------------------------------------------------------------------------------

# plt.pcolor(z.detach().numpy(),cmap='inferno')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.colorbar()
# plt.show()


# fig, axs = plt.subplots(3)

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

# plt.show()
#-----------------------------------------------------------------------------------------
#Visualizing-----------------------------------------------------------------------------

# def visualize(c,predict):
#     fig = plt.figure()
#     ax = plt.axes(ylim=(0,1),xlim=(0,200))

#     line1, = ax.plot([],[],label='original')
#     line2, = ax.plot([],[],label='prediction')

#     def init():
#         line1.set_data([],[])
#         line2.set_data([],[])
#         return line1, line2


#     def animate(i):
#         print(i)
#         line1.set_data(np.arange(200),c[i])
#         line2.set_data(np.arange(200),predict[i])
#         return line1, line2

#     anim = animation.FuncAnimation(
#                                    fig, 
#                                    animate, 
#                                    init_func = init,
#                                    frames = 200,
#                                    interval = 200,
#                                    blit = True
#                                    )

#     ax.legend()
#     plt.show()
#-----------------------------------------------------------------------------------------
#Bad Mistakes----------------------------------------------------------------------------
mistake_list = []
for i in range(5000):
    mistake = np.sum(np.abs(c[i] - predict[i]))
    mistake_list.append((i,mistake))

zip(mistake_list)

# plt.plot(c[900],'-o''m',label='$Original$')
# plt.plot(predict[900],'-v''k',label='$Prediction$')
# plt.xlabel('$Velocity$')
# plt.ylabel('$Probability$')
# plt.legend()
# plt.show()

# np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_500_c.txt',c[500])
# np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_500_p.txt',predict[500])
# np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_Samples_1_1_lin.txt',mistake_list)
#theta = np.linspace(0.0,2*np.pi,5000,endpoint=False)
#width = (2*np.pi) / 5000
# ax = plt.subplot(111, polar=False)
# bars = ax.bar(range(len(mistake_list)),[val[1]for val in mistake_list],color='k',width=1)
# axr = ax.twiny()    
# axr.xaxis.set_major_locator(plt.FixedLocator(np.arange(0,25)))
# axr.set_xlim((0,25))
# ax.set_xlim((0,2499))
# ax.yaxis.grid(True)
# axr.xaxis.grid(True)
# ax.set_xlabel(r'$Samples$')
# axr.set_xlabel(r'$Timesteps$')
# ax.set_ylabel(r'$Absolute Error$')
# plt.show()
#-------------------------------------------------------------------------------------------
#Visualizing Density-----------------------------------------------------------------------
# def density(c,predict):

#     rho_predict = np.zeros([25,200])
#     rho_samples = np.zeros([25,200])
#     n=0

#     for k in range(25):
#         for i in range(200):
#             rho_samples[k,i] = np.sum(c[i+n]) * 0.5128
#             rho_predict[k,i] = np.sum(predict[i+n]) * 0.5128  
#         n += 200
#     return rho_samples, rho_predict

# rho_s, rho_p = density(c,predict)

# visualize(rho_s,rho_p)

# print('Verage Density Error', np.sum(np.abs(rho_s - rho_p))/len(rho_s))
# print('Average Test Error', np.sum(np.abs(c - predict))/len(c))

# plt.plot(np.linspace(0,1,200),rho_s[-1],'-o''k',label='$Original$')
# plt.plot(np.linspace(0,1,200),rho_p[-1],'-v''k',label='$Prediction$')
# plt.legend()
# plt.xlabel('$Space$')
# plt.ylabel('$Density$')
# plt.show()
# -------------------------------------------------------------------------------------------
f = c
rec = predict
ph_error = LA.norm((f - rec).flatten())/LA.norm(f.flatten())

print(ph_error)






