'''
Plot results Linear
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
import torch.nn as nn
import scipy.io as sio
import torch.tensor as tensor
import matplotlib.animation as animation
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':15})

# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


INPUT_DIM = 40
HIDDEN_DIM = 20
LATENT_DIM = 5


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(in_features=input_dim, 
                                    out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, 
                                    out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, 
                                    out_features=lat_dim)
        self.activation_out = nn.LeakyReLU()
    def forward(self, x):
        x = self.activation_out(self.linear1(x))
        x = self.activation_out(self.linear2(x))
        x = self.activation_out(self.linear3(x))

        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Decoder, self).__init__()

        self.linear4 = nn.Linear(in_features=lat_dim, 
                                    out_features=hidden_dim)
        self.linear5 = nn.Linear(in_features=hidden_dim, 
                                    out_features=hidden_dim)
        self.linear6 = nn.Linear(in_features=hidden_dim, 
                                    out_features=input_dim)
        self.activation_out = nn.LeakyReLU()

    def forward(self,x):
        x = self.activation_out(self.linear4(x))
        x = self.activation_out(self.linear5(x))
        x = self.activation_out(self.linear6(x))
      
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
encoder = Encoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder)


model.load_state_dict(torch.load('Lin_AE_STATE_DICT.pt',map_location='cpu'))
model.eval()

# load original data
f = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/f.mat')
f = f['f']

x=200
t=25
v=40

#Submatrix
c = np.zeros((t*x,v))
n = 0

#Build 2D-Version
for i in range(t):                                             # T (zeilen)
    for j in range(v):                                         # V (spalten)
            c[n:n+x,j]=f[i,j,:]

    n += x


#Inference

c = tensor(c, dtype=torch.float)


predict, z = model(c)
c = c.detach().numpy()
predict = predict.detach().numpy()


# # plot code

# print(z.shape)
# plt.plot(np.arange(5000),z[:,0].detach().numpy())
# plt.show()

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


# #Bad Mistakes

# mistake_list = []
# for i in range(4999):
#     mistake = np.sum(np.abs(c[i] - predict[i]))
#     mistake_list.append(mistake)


# index=mistake_list.index(np.max(mistake_list))


# plt.plot(c[500])
# plt.plot(predict[500])
# plt.show()


# np.save('/home/zachary/Desktop/BA/Plotting_Data/Mistakes_Samples_1_1_lin',mistake_list)

# plt.bar(np.arange(4999),mistake_list,label='$Absolute Error$')
# plt.legend()
# plt.xlabel('$Samples$')
# plt.ylabel('$Absolute Error$')
# plt.grid()
# plt.tight_layout(pad=0.2)
# plt.show()

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

plt.plot(rho_s[-1])
plt.plot(rho_p[-1])
plt.show()