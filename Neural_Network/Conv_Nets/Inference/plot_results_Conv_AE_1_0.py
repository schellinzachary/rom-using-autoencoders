'''
Plot results Convolutional AE 1.0 '''
import numpy as np
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
plt.rcParams['ytick.labelsize']=15

def net(c):

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.convE1 = nn.Conv2d(1,8,(3,2),stride=2)
            self.convE2 = nn.Conv2d(8,16,(2,2),stride=2)
            self.convE3 = nn.Conv2d(16,32,(2,2),stride=2)
            self.convE4 = nn.Conv2d(32,64,(3,3),stride=(1,2))
            self.linearE1 = nn.Linear(in_features=768,out_features=3)
            self.act = nn.Tanh()

        def forward(self, x):
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            x = self.act(self.convE3(x))
            x = self.act(self.convE4(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=3, out_features=768)
            self.convD1 = nn.ConvTranspose2d(64,32,(3,3),stride=(1,2))
            self.convD2 = nn.ConvTranspose2d(32,16,(2,2),stride=2)
            self.convD3 = nn.ConvTranspose2d(16,8,(2,2),stride=2)
            self.convD4 = nn.ConvTranspose2d(8,1,(3,2),stride=2)
            self.act = nn.Tanh()

        def forward(self, x):
            x = self.linearD1(x)
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
            z = self.encoder(x)
            x = self.decoder(z)
            return x, z



    #encoder
    encoder = Encoder()

    #decoder
    decoder = Decoder()

    #Autoencoder
    model = Autoencoder(encoder, decoder)




    checkpoint = torch.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Conv_Nets/Conv_State_Dicts/Conv_AE_STATE_DICT_0_9_L3_B4_lr-4_Tanh.pt')

    model.load_state_dict(checkpoint['model_state_dict'])
    #model.eval()


    #-------------------------------------------------------------------------------------------
    #Inference---------------------------------------------------------------------------------
    c = tensor(c, dtype=torch.float)

    predict,z = model(c)
    c = c.detach().numpy()
    predict = predict.detach().numpy()


    return predict, z

# load original data-----------------------------------------------------------------------
c = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/preprocessed_samples_conv.npy')


#Inference-----------------------------------------------------------------------------------
predict, z = net(c)

#------------------------------------------------------------------------------------------
# plot code-------------------------------------------------------------------------------
fig, axs = plt.subplots(5)

axs[0].plot(np.arange(40),z[:,0].detach().numpy(),'k')
axs[0].set_ylabel('#')
axs[0].set_xlabel('x')

axs[1].plot(np.arange(40),z[:,1].detach().numpy(),'k')
axs[1].set_ylabel('#')
axs[1].set_xlabel('x')

axs[2].plot(np.arange(40),z[:,2].detach().numpy(),'k')
axs[2].set_ylabel('#')
axs[2].set_xlabel('x')

plt.show()
#-----------------------------------------------------------------------------------------
#Visualizing-----------------------------------------------------------------------------
c = np.squeeze(c,axis=1)
predict = np.squeeze(predict,axis=1)

#plt.imshow(c[37])
plt.imshow(predict[37])
plt.colorbar()
plt.show()
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
for i in range(40):
    mistake = np.sum(np.abs(c[i] - predict[i]),axis=None)/(200*25)
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
ax = plt.subplot(111, polar=False)
bars = ax.bar(range(len(mistake_list)),[val[1]for val in mistake_list],color='k',width=1)
axr = ax.twiny()    
axr.xaxis.set_major_locator(plt.FixedLocator(np.arange(0,25)))
axr.set_xlim((0,25))
ax.set_xlim((0,40))
ax.yaxis.grid(True)
axr.xaxis.grid(True)
ax.set_xlabel(r'$Samples$')
axr.set_xlabel(r'$Timesteps$')
ax.set_ylabel(r'$Absolute Error$')
plt.show()
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
test_error = np.sum(np.abs(c - predict),axis=None)
mean = np.sum(test_error)/(200*25*40)
print('Mean Test Error', mean)
print('STD Test Error', ((1/(len(test_error)-1)) * np.sum((test_error - mean)**2 )))
print('Abweichung vom Mean',np.sum(np.abs(test_error - mean)) / len(test_error))
print('Highest Sample Error',np.max(test_error))
print('Lowest Sample Error', np.min(test_error))







