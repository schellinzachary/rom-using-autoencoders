import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils import data
from numpy import pi, linspace, sin, zeros
import matplotlib.pyplot as plt
import scipy.io as sio

#load data
f1 = sio.loadmat('/home/zachary/BA/data_sod/sod241Kn0p00001/f.mat')
f1 = f1['f']
NUM_SAMPLES     = f1.shape[0] 
NUM_VELOCITY    = f1.shape[1] 
NUM_SPACE       = f1.shape[2]
rho = np.zeros([NUM_SAMPLES,NUM_SPACE])
for i in range(NUM_SAMPLES):
    for k in range(NUM_SPACE):
        f1[i,:,k] = (f1[i,:,k] - np.amin(f1[i,:,k])) / (np.amax(f1[i,:,k])-np.amin(f1[i,:,k]))



samples = f1.reshape(NUM_SAMPLES,NUM_SPACE*NUM_VELOCITY)
np.savetxt('Norm.out', samples, delimiter=',')


N_TRAIN_STEPS = 20000




class classify(nn.Module):
    def __init__(self):
        # initialize as nn.Module
        super(classify, self).__init__()

        
        self.linear1 = nn.Linear(in_features=NUM_SPACE*NUM_VELOCITY, 
                                    out_features=100)
        self.linear2 = nn.Linear(in_features=100, 
                                    out_features=100)
        self.linear3 = nn.Linear(in_features=100, 
                                    out_features=5)
        self.linear4 = nn.Linear(in_features=5, 
                                    out_features=100)
        self.linear5 = nn.Linear(in_features=100, 
                                    out_features=100)
        self.linear6 = nn.Linear(in_features=100, 
                                    out_features=NUM_SPACE*NUM_VELOCITY)
        self.activation_out = nn.LeakyReLU(negative_slope=0.001)

    def forward(self, x):
        x = self.activation_out(self.linear1(x))
        x = self.activation_out(self.linear2(x))
        x = self.activation_out(self.linear3(x))
        x = self.activation_out(self.linear4(x))
        x = self.activation_out(self.linear5(x))
        output1 = self.activation_out(self.linear6(x))
      
        return output1.squeeze()


if __name__ == '__main__':

    classify = classify()

    train_ds_in = tensor(samples,
                         dtype=torch.float)
    
    optimizer = Adam(params=classify.parameters(), lr=0.001)

    loss_crit = nn.L1Loss()
    train_losses = []

    plt.ion()
    plt.figure()  

    for step in range(N_TRAIN_STEPS):
    	

        cl_out = classify(train_ds_in)

        train_loss = loss_crit(cl_out, train_ds_in)
        
        train_losses.append(train_loss.item())

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        print('Epoch :',step, 'train_loss:',train_loss,':)')

        if train_loss <= 0.0013:
        	break

 
        plt.semilogy(np.arange(step+1), train_losses, label='Training loss')
        plt.legend(loc='upper right')
        plt.xlabel('trainstep')
        plt.ylabel('loss')
        plt.draw()
        plt.pause(0.001)
        plt.clf()

samples = tensor(samples, dtype=torch.float)

predict = classify(samples)

samples  = samples.detach().numpy()
samples = samples.reshape(NUM_SAMPLES,NUM_VELOCITY,NUM_SPACE)
predict = predict.detach().numpy()
predict = predict.reshape(NUM_SAMPLES,NUM_VELOCITY,NUM_SPACE)

rho_predict = np.zeros([NUM_SAMPLES,NUM_SPACE])
rho_samples = np.zeros([NUM_SAMPLES,NUM_SPACE])
for i in range(NUM_SAMPLES):
    for k in range(NUM_SPACE):
        rho_samples[i,k] = np.sum(samples[i,:,k]) * 0.5128
        rho_predict[i,k] = np.sum(predict[i,:,k]) * 0.5128
        

plt.ion()
plt.figure()
for i in range(NUM_SAMPLES):   
    plt.plot(rho_predict[i,:])
    plt.plot(rho_samples[i,:])
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('A * sin(g*x + t)')
    plt.draw()
    plt.pause(0.001)
    plt.clf()
