'''
Linear Autoencoder v1.1
'''



import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load data
f1 = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/f.mat')
f1 = f1['f']

NUM_SAMPLES     = f1.shape[0] 
NUM_VELOCITY    = f1.shape[1] 
NUM_SPACE       = f1.shape[2]

BATCH_SIZE = 5
lr = 1e-4

rho = np.zeros([NUM_SAMPLES,NUM_SPACE])
for i in range(NUM_SAMPLES):
    for k in range(NUM_SPACE):
        f1[i,:,k] = (f1[i,:,k] - np.amin(f1[i,:,k])) / (np.amax(f1[i,:,k])-np.amin(f1[i,:,k]))


f1 = tensor(f1, dtype=torch.float).to(device)
f1 = torch.reshape(f1,(NUM_SAMPLES,NUM_SPACE*NUM_VELOCITY))
train_iterator = DataLoader(f1, batch_size = BATCH_SIZE, shuffle=True)



N_TRAIN_STEPS = 1000
t = 500


class Autoencoder(nn.Module):
    def __init__(self):
        # initialize as nn.Module
        super(Autoencoder, self).__init__()

        self.linear1 = nn.Linear(in_features=NUM_SPACE*NUM_VELOCITY, 
                                    out_features=t)
        self.linear2 = nn.Linear(in_features=t, 
                                    out_features=t)
        self.linear3 = nn.Linear(in_features=t, 
                                    out_features=5)
        self.linear4 = nn.Linear(in_features=5, 
                                    out_features=t)
        self.linear5 = nn.Linear(in_features=t, 
                                    out_features=t)
        self.linear6 = nn.Linear(in_features=t, 
                                    out_features=NUM_SPACE*NUM_VELOCITY)
        self.activation_out = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.activation_out(self.linear1(x))
        x = self.activation_out(self.linear2(x))
        #x = self.activation_out(self.linear3(x))
        #x = self.activation_out(self.linear4(x))
        x = self.activation_out(self.linear5(x))
        output1 = self.activation_out(self.linear6(x))
      
        return output1.squeeze()


if __name__ == '__main__':

    Autoencoder = Autoencoder().to(device)
    
    optimizer = Adam(params=Autoencoder.parameters(), lr=lr)

    loss_crit = nn.L1Loss()
    train_losses = []



    for step in range(N_TRAIN_STEPS):
    	
        for batch_ndx, x in enumerate(train_iterator):

            x_out = Autoencoder(x).to(device)

            train_loss = loss_crit(x_out, x)

            train_loss /= len(train_iterator)
            train_losses.append(train_loss.item())

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

        
        print('Epoch :',step, 'train_loss:',train_loss,':)')



plt.figure()
plt.semilogy(np.arange(5*step+5), train_losses, label='Training loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()



f1 = tensor(f1, dtype=torch.float).to(device)

predict = Autoencoder(f1).to(device)

f1 = f1.to('cpu')
predict= predict.to('cpu')

f1  = f1.detach().numpy()
f1 = f1.reshape(NUM_SAMPLES,NUM_VELOCITY,NUM_SPACE)

predict = predict.detach().numpy()
predict = predict.reshape(NUM_SAMPLES,NUM_VELOCITY,NUM_SPACE)


rho_predict = np.zeros([NUM_SAMPLES,NUM_SPACE])
rho_f1 = np.zeros([NUM_SAMPLES,NUM_SPACE])
for i in range(NUM_SAMPLES):
    for k in range(NUM_SPACE):
        rho_f1[i,k] = np.sum(f1[i,:,k]) * 0.5128
        rho_predict[i,k] = np.sum(predict[i,:,k]) * 0.5128
        


plt.ion()
plt.figure()
for i in range(NUM_SAMPLES):   
    plt.plot(rho_predict[i,:])
    plt.plot(rho_f1[i,:])
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('A * sin(g*x + t)')
    plt.draw()
    plt.pause(0.001)
    plt.clf()
