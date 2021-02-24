import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta, Adagrad
import torch.tensor as tensor
from torch.utils import data
from numpy import pi, linspace, sin, zeros
import matplotlib.pyplot as plt
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load data
f1 = sio.loadmat('/home/bapu/BA/data_sod/sod241Kn0p00001/f.mat')
f1 = f1['f']
NUM_SAMPLES     = f1.shape[0] 
NUM_VELOCITY    = f1.shape[1] 
NUM_SPACE       = f1.shape[2]


rho = np.zeros([NUM_SAMPLES,NUM_SPACE])
for i in range(NUM_SAMPLES):
    for k in range(NUM_SPACE):
        f1[i,:,k] = (f1[i,:,k] - np.amin(f1[i,:,k])) / (np.amax(f1[i,:,k])-np.amin(f1[i,:,k]))


f2 = np.copy(f1)
np.random.shuffle(f1)
samples = f1.reshape(NUM_SAMPLES,NUM_SPACE*NUM_VELOCITY)
train_list  = ('train' + str(i) for i in range(20))
train_list  = np.split(samples[1:NUM_SAMPLES,:],20,axis=0)


N_TRAIN_STEPS = 20000
t = 100

class classify(nn.Module):
    def __init__(self):
        # initialize as nn.Module
        super(classify, self).__init__()

        
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
        #x = self.activation_out(self.linear2(x))
        x = self.activation_out(self.linear3(x))
        x = self.activation_out(self.linear4(x))
        #x = self.activation_out(self.linear5(x))
        output1 = self.activation_out(self.linear6(x))
      
        return output1.squeeze()


if __name__ == '__main__':

    classify = classify().to(device)
    
    optimizer = Adam(params=classify.parameters(), lr=0.001)

    loss_crit = nn.L1Loss()
    train_losses = []

    plt.ion()
    plt.figure()  

    for step in range(N_TRAIN_STEPS):
    	
        for p in range(20):

            train_in = train_list[p]

            train_in = tensor(train_in,
                         dtype=torch.float).to(device)

            cl_out = classify(train_in).to(device)

            train_loss = loss_crit(cl_out, train_in)
            
            train_losses.append(train_loss.item())

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

        print('Epoch :',step, 'train_loss:',train_loss,':)')

        if train_loss <= 0.0003:
            	break

 
        plt.semilogy(np.arange(20*step+20), train_losses, label='Training loss')
        plt.legend(loc='upper right')
        plt.xlabel('trainstep')
        plt.ylabel('loss')
        plt.draw()
        plt.pause(0.001)
        plt.clf()


f1 = f2.reshape(NUM_SAMPLES,NUM_SPACE*NUM_VELOCITY)
f1 = tensor(f1, dtype=torch.float).to(device)

predict = classify(f1).to(device)

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
        

np.save('/home/bapu/BA/Neural_Network/Results/rho_predict_lin',rho_predict)
np.save('/home/bapu/BA/Neural_Network/Results/rho_samples_lin',rho_samples)
np.save('/home/bapu/BA/Neural_Network/Results/losses_lin',train_losses)
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
