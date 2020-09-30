'''
Plot results CAE 1.0
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor
import matplotlib.animation as animation
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':25})

# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)



# load original data
c = np.load('/home/fusilly/ROM_using_Autoencoders/data_sod/original_data_in_format.npy')
c = c.T


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
                                        out_features=lat_dim, bias=False)
            self.activation_out = nn.LeakyReLU()
            self.activation_out1 = nn.LeakyReLU()
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
            h = self.enc(x)
            predicted = self.dec(h)
            return predicted



    class Autoencoder(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, x):
            h = self.enc(x)
            predicted = self.dec(h)
            return predicted, h

    #encoder
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    #decoder
    decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    #Autoencoder
    model = Autoencoder(encoder, decoder)


    model.load_state_dict(torch.load('CAE_STATE_DICT_1_0_L5_16_substr50_tiedWeights_LR.pt',map_location='cpu'))
    model.eval()



    #Inference

    c = tensor(c, dtype=torch.float)


    predict, z = model(c)
    c = c.detach().numpy()
    predict = predict.detach().numpy()




    W = encoder.state_dict()['linear2.weight']
    dh = torch.where(W >= 0 , torch.ones(1), torch.ones(1)*1e-2) 
    j = W * dh
    # dh = z.detach().numpy() * (1 - z.detach().numpy())

    # j = np.matmul(dh,W)

    u, s, vh = np.linalg.svd(j.detach().numpy(),full_matrices=False) #s Singularvalues

    plt.plot(np.arange(s.shape[0]),s,'*')
    plt.show()

    # # plot code

    fig, axs = plt.subplots(5)
    axs[0].plot(np.arange(5000),z[:,0].detach().numpy())
    axs[1].plot(np.arange(5000),z[:,1].detach().numpy())
    axs[2].plot(np.arange(5000),z[:,2].detach().numpy())
    axs[3].plot(np.arange(5000),z[:,3].detach().numpy())
    axs[4].plot(np.arange(5000),z[:,4].detach().numpy())
    # axs[5].plot(np.arange(5000),z[:,5].detach().numpy())
    plt.show()
    return predict


# #Visualizing

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


predict = net(c)


# #Bad Mistakes

mistake_list = []
for i in range(4999):
    mistake = np.sum(np.abs(c[i] - predict[i]))
    mistake_list.append((i,mistake))

zip(mistake_list)


#index=mistake_list.index(np.max(mistake_list[0,:]))


plt.plot(c[501],'-o''m',label='$Original$')
plt.plot(predict[501],'-v''k',label='$Prediction$')
plt.xlabel('$Velocity$')
plt.ylabel('$Probability$')
plt.legend()
plt.show()

# np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_500_c.txt',c[500])
# np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_500_p.txt',predict[500])
# np.savetxt('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_Samples_1_1_lin.txt',mistake_list)

plt.bar(range(len(mistake_list)),[val[1]for val in mistake_list],color='k')
plt.xlabel('$Samples$')
plt.ylabel('$Absolute Error$')
plt.grid()
plt.tight_layout()
plt.show()

#Visualizing Density

def density(c,predict):

    rho_predict = np.zeros([25,200])
    rho_samples = np.zeros([25,200])
    n=0

    for k in range(25):
        for i in range(200):
            rho_samples[k,i] = np.sum(c[i+n]) * 0.5128
            rho_predict[k,i] = np.sum(predict[i+n]) * 0.5128   
        n += 200
    return rho_samples, rho_predict

rho_s, rho_p = density(c,predict)

visualize(rho_s,rho_p)

print('mis_rho', np.sum(np.abs(rho_s - rho_p))/len(rho_s))
print('mis_samples',np.sum(np.abs(c - predict))/len(c))

plt.plot(np.linspace(0,1,200),rho_s[-1],'-o''m',label='$Original$')
plt.plot(np.linspace(0,1,200),rho_p[-1],'-v''k',label='$Prediction$')
plt.legend()
plt.xlabel('$Space$')
plt.ylabel('$Density$')
plt.show()