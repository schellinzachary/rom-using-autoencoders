'''
Plot results Convolutional AE 1.0 '''
import numpy as np
from numpy.linalg import norm as norm
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
code = 32

def net(c):
    
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.convE1 = nn.Conv2d(1,8,(5,10),stride=(4,5))
            self.convE2 = nn.Conv2d(8,16,(2,5),stride=(2,5))
            self.linearE1 = nn.Linear(in_features=256,out_features=i)
            self.act = nn.Tanh()
            #self.act_c = nn.Tanh()

        def forward(self, x):
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            #x = self.act_c(self.linearE1(x))
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=i, out_features=448)
            self.convD1 = nn.ConvTranspose2d(16,8,(2,5),stride=(2,5))
            self.convD2 = nn.ConvTranspose2d(8,1,(5,10),stride=(4,5))
            self.act = nn.Tanh()
            #self.act_c = nn.Tanh()

        def forward(self, x):
            x = self.linearD1(x)
            #x = self.act_c(self.linearD1(x))
            dim = x.shape[0]
            x = torch.reshape(x,[dim,16,2,8])
            x = self.act(self.convD1(x))
            x = self.act(self.convD2(x))
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

    checkpoint = torch.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Conv_Nets/Code/1_1/CoAE_SD_1_32.pt')

    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    N_EPOCHS = checkpoint['epoch']
    print(N_EPOCHS)
    #model.eval()


    #-------------------------------------------------------------------------------------------
    #Inference---------------------------------------------------------------------------------
    c = tensor(c, dtype=torch.float)

    predict,z = model(c)
    c = c.detach().numpy()
    predict = predict.detach().numpy()


    return predict, z

# load original data-----------------------------------------------------------------------
c = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Preprocessing/preprocessed_samples_conv_unshuffled.npy')


#Inference-----------------------------------------------------------------------------------
predict, z = net(c)

#------------------------------------------------------------------------------------------
# plot code-------------------------------------------------------------------------------
for i in range(code):
    print(f'#%s Code'%i,np.sum(np.abs(z[:,i].detach().numpy())))
fig, axs = plt.subplots(3)

axs[0].plot(np.arange(40),z[:,0].detach().numpy(),'k')
axs[0].set_ylabel('#')
axs[0].set_xlabel('x')

axs[1].plot(np.arange(40),z[:,1].detach().numpy(),'k')
axs[1].set_ylabel('#')
axs[1].set_xlabel('x')

# axs[2].plot(np.arange(40),z[:,2].detach().numpy(),'k')
# axs[2].set_ylabel('#')
# axs[2].set_xlabel('x')


#-----------------------------------------------------------------------------------------
#Visualizing-----------------------------------------------------------------------------
c = np.squeeze(c,axis=1)
predict = np.squeeze(predict,axis=1)

fig1, axs1 = plt.subplots(nrows=2)
a = 37
org = axs1[0].imshow(c[a],vmin=0, vmax=np.max(c[a]))
pred = axs1[1].imshow(predict[a],vmin=0,vmax=np.max(c[a]))
fig1.colorbar(org, ax = axs1[0])
fig1.colorbar(pred, ax = axs1[1])
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

#-----------------------------------------------------------------------------------------
#Bad Mistakes----------------------------------------------------------------------------
# mistake_list = []
# for i in range(40):
#     mistake = np.sum(np.abs(c[i] - predict[i]),axis=None)/(200*25)
#     mistake_list.append((i,mistake))

# zip(mistake_list)

# ax = plt.subplot(111, polar=False)
# bars = ax.bar(range(len(mistake_list)),[val[1]for val in mistake_list],color='k')
# #ax.set_xlim((0,39))
# #ax.yaxis.grid(True)
# ax.xaxis.grid(True, which='major')
# ax.set_xlabel(r'$Samples$')
# ax.set_ylabel(r'$Absolute Error$')

#-------------------------------------------------------------------------------------------
#Visualizing Density-----------------------------------------------------------------------
def density(c,predict):
    c = np.swapaxes(c,0,1)
    predict = np.swapaxes(predict,0,1)

    rho_predict = np.zeros([25,200])
    rho_samples = np.zeros([25,200])
    n=0

    for k in range(25):
        for i in range(200):
            rho_samples[k,i] = np.sum(c[k,:,i]) * 0.5128
            rho_predict[k,i] = np.sum(predict[k,:,i]) * 0.5128  
    return rho_samples, rho_predict



rho_s, rho_p = density(c,predict)

#visualize(rho_s,rho_p)

print('Verage Density Error', np.sum(np.abs(rho_s - rho_p))/len(rho_s))
print('Average Test Error', np.sum(np.abs(c - predict))/len(c))

plt.plot(np.linspace(0,1,200),rho_s[-1],'-o''k',label='$Original$')
plt.plot(np.linspace(0,1,200),rho_p[-1],'-v''k',label='$Prediction$')
plt.legend()
plt.xlabel('$Space$')
plt.ylabel('$Density$')
plt.show()
# # -------------------------------------------------------------------------------------------
test_error = norm((c[:] - predict[:]).flatten())/norm(c[:].flatten())
print(test_error)
#test_error = np.sum(np.abs(c - predict),axis=None)
#mean = np.sum(test_error)/(200*25*40)
#print('Mean Test Error', mean)
#print('STD Test Error', ((1/(len(test_error)-1)) * np.sum((test_error - mean)**2 )))
# print('Abweichung vom Mean',np.sum(np.abs(test_error - mean)) / len(test_error))
# print('Highest Sample Error',np.max(test_error))
# print('Lowest Sample Error', np.min(test_error))
plt.show()






