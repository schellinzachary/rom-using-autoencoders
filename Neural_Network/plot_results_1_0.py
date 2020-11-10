'''
Plot results Linear 1.0
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor
import matplotlib.animation as animation
from scipy.interpolate import interp1d
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':15})

# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

def net(c):

    INPUT_DIM = 40
    HIDDEN_DIM = 20
    LATENT_DIM = 5


    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, lat_dim):
            super(Encoder, self).__init__()

            self.linear1 = nn.Linear(in_features=input_dim, 
                                        out_features=hidden_dim)
            self.linear2 = nn.Linear(in_features=hidden_dim, 
                                        out_features=lat_dim)
            self.activation_out = nn.LeakyReLU()
            self.activation_out1 = nn.Tanh()
            #self.activation_out1 = nn.LeakyReLU()
            #self.activation_out1 = nn.Sigmoid()
        def forward(self, x):
            x = self.activation_out(self.linear1(x))
            x = self.activation_out1(self.linear2(x))
            return x


    class Decoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, lat_dim):
            super(Decoder, self).__init__()
            self.linear3 = nn.Linear(in_features=lat_dim, 
                                    out_features=hidden_dim)
            self.linear4 = nn.Linear(in_features=hidden_dim, 
                                    out_features=input_dim)
            self.activation_out = nn.LeakyReLU()

        def forward(self,x):
            x = self.activation_out(self.linear3(x))
            x = self.activation_out(self.linear4(x))
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

    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


    #encoder
    encoder = Encoder(INPUT_DIM,HIDDEN_DIM, LATENT_DIM)

    #decoder
    decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    #Autoencoder
    model = Autoencoder(encoder, decoder)




    model.load_state_dict(torch.load('Lin_AE_STATE_DICT_1_0_L5_16_lr-3_TH_second.pt',map_location='cpu'))
    model.eval()

    W = encoder.state_dict()['linear2.weight']

    #-------------------------------------------------------------------------------------------
    #Inference---------------------------------------------------------------------------------
    c = tensor(c, dtype=torch.float)

    predict,z = model(c)
    c = c.detach().numpy()
    predict = predict.detach().numpy()

    return predict, W, z

# load original data-----------------------------------------------------------------------
c = np.load('/home/fusilly/ROM_using_Autoencoders/data_sod/original_data_in_format.npy')
c = c
c = c.T
c_half = c[::2]

#Inference-----------------------------------------------------------------------------------
predict, W, z = net(c_half)
#-------------------------------------------------------------------------------------------
# Jacobian---------------------------------------------------------------------------------
#dh = torch.where(W >= 0 , torch.ones(1), torch.ones(1)*1e-2)
# dh = 1 - z**2 
# j = torch.mm(dh,W)
# print(j.shape)
# u, s, vh = np.linalg.svd(j.detach().numpy(),full_matrices=False) #s Singularvalues

# plt.plot(np.arange(1,6),s,'-*''k')
# plt.ylabel(r'#',fontsize=25)
# plt.xlabel(r'Singular Value',fontsize=25)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.show()
#------------------------------------------------------------------------------------------
# plot code-------------------------------------------------------------------------------
# fig, axs = plt.subplots(5)

# axs[0].plot(np.arange(2500),z[:,0].detach().numpy(),'k')
# axs[0].set_ylabel('#')
# axs[0].set_xlabel('x')

# axs[1].plot(np.arange(2500),z[:,1].detach().numpy(),'k')
# axs[1].set_ylabel('#')
# axs[1].set_xlabel('x')

# axs[2].plot(np.arange(2500),z[:,2].detach().numpy(),'k')
# axs[2].set_ylabel('#')
# axs[2].set_xlabel('x')

# axs[3].plot(np.arange(2500),z[:,3].detach().numpy(),'k')
# axs[3].set_ylabel('#')
# axs[3].set_xlabel('x')

# axs[4].plot(np.arange(2500),z[:,4].detach().numpy(),'k')
# axs[4].set_ylabel('#')
# axs[4].set_xlabel('x')

# plt.show()
#-----------------------------------------------------------------------------------------
#Visualizing-----------------------------------------------------------------------------

def visualize(c,predict):
    fig = plt.figure()
    ax = plt.axes(ylim=(0,1),xlim=(0,200))

    line1, = ax.plot([],[],label='original')
    line2, = ax.plot([],[],label='prediction')

    def init():
        line1.set_data([],[])
        line2.set_data([],[])
        return line1, line2


    def animate(i):
        print(i)
        line1.set_data(np.arange(200),c[i])
        line2.set_data(np.arange(200),predict[i])
        return line1, line2

    anim = animation.FuncAnimation(
                                   fig, 
                                   animate, 
                                   init_func = init,
                                   frames = 200,
                                   interval = 200,
                                   blit = True
                                   )

    ax.legend()
    plt.show()
#-----------------------------------------------------------------------------------------
#Bad Mistakes----------------------------------------------------------------------------
mistake_list = []
for i in range(2499):
    mistake = np.sum(np.abs(c_half[i] - predict[i]))
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
test_error = np.sum(np.abs(c_half - predict),axis=1)
mean = np.sum(test_error)/len(test_error)
print('Mean Test Error', mean)
print('STD Test Error', ((1/(len(test_error)-1)) * np.sum((test_error - mean)**2 )))
print('Abweichung vom Mean',np.sum(np.abs(test_error - mean)) / len(test_error))
print('Highest Sample Error',np.max(test_error))
print('Lowest Sample Error', np.min(test_error))

#--------------------------------------------------------------------------------------------
#Interpolate
x = np.linspace(1, 2500, num=2500, endpoint=True)
x_new = np.linspace(1,2500, num=5000, endpoint=True)
I_1 = interp1d(x, z[:,0].detach().numpy(), kind='linear')
I_2 = interp1d(x, z[:,1].detach().numpy(), kind='linear')
I_3 = interp1d(x, z[:,2].detach().numpy(), kind='linear')
I_4 = interp1d(x, z[:,3].detach().numpy(), kind='linear')
I_5 = interp1d(x, z[:,4].detach().numpy(), kind='linear')

z_1 = I_1(x_new)
z_2 = I_2(x_new)
z_3 = I_3(x_new)
z_4 = I_4(x_new)
z_5 = I_5(x_new)

z_new = np.vstack((z_1,z_2,z_3,z_4,z_5))
z_new = z_new.T

def decoder(c):

    INPUT_DIM = 40
    HIDDEN_DIM = 20
    LATENT_DIM = 5


    class Decoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, lat_dim):
            super(Decoder, self).__init__()
            self.linear3 = nn.Linear(in_features=lat_dim, 
                                    out_features=hidden_dim)
            self.linear4 = nn.Linear(in_features=hidden_dim, 
                                    out_features=input_dim)
            self.activation_out = nn.LeakyReLU()

        def forward(self,x):
            x = self.activation_out(self.linear3(x))
            x = self.activation_out(self.linear4(x))
            return x


    class Autoencoder(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, x):
            predicted = self.dec(c)
            return predicted

    #decoder
    decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    #Autoencoder
    model = Autoencoder(decoder)



    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict) 
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)

    model.load_state_dict(torch.load('Lin_AE_STATE_DICT_1_0_L5_16_lr-3_TH_second.pt',map_location='cpu'),strict=False)
    model.eval()


    #-------------------------------------------------------------------------------------------
    #Inference---------------------------------------------------------------------------------
    c = tensor(c, dtype=torch.float)

    predict = model(c)
    c = c.detach().numpy()
    predict = predict.detach().numpy()

    return predict

predict_new = decoder(z_new)
#-----------------------------------------------------------------------------------------------
#Test Interpolation
test_error = np.sum(np.abs(c - predict_new),axis=1)
mean = np.sum(test_error)/len(test_error)
print('Mean Test Error', mean)
print('STD Test Error', ((1/(len(test_error)-1)) * np.sum((test_error - mean)**2 )))
print('Abweichung vom Mean',np.sum(np.abs(test_error - mean)) / len(test_error))
print('Highest Sample Error',np.max(test_error))
print('Lowest Sample Error', np.min(test_error))

#Bad Mistakes----------------------------------------------------------------------------
mistake_list = []
for i in range(5000):
    mistake = np.sum(np.abs(c[i] - predict_new[i]))
    mistake_list.append((i,mistake))

zip(mistake_list)

plt.plot(c[399],'-o''m',label='$Original$')
plt.plot(predict_new[399],'-v''k',label='$Prediction$')
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
ax.set_xlim((0,5000))
ax.yaxis.grid(True)
axr.xaxis.grid(True)
ax.set_xlabel(r'$Samples$')
axr.set_xlabel(r'$Timesteps$')
ax.set_ylabel(r'$Absolute Error$')
plt.show()






