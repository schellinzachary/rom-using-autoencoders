'''
Convolution Autoencoder v1.1
'''


import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta, Adagrad
import torch.tensor as tensor
import matplotlib.pyplot as plt
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load data
f1 = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/f.mat')
f1 = f1['f']
NUM_SAMPLES     = f1.shape[0] 
NUM_VELOCITY    = f1.shape[1] 
NUM_SPACE       = f1.shape[2]


rho = np.zeros([NUM_SAMPLES,NUM_SPACE])
for i in range(NUM_SAMPLES):
    for k in range(NUM_SPACE):
        f1[i,:,k] = (f1[i,:,k] - np.amin(f1[i,:,k])) / (np.amax(f1[i,:,k])-np.amin(f1[i,:,k]))

f1 = np.expand_dims(f1, axis=1)
f2 = np.copy(f1)
np.random.shuffle(f1)
samples = f1
train_list  = np.split(samples[0:NUM_SAMPLES,:,:,:],5,axis=0)
train_list = tensor(train_list,dtype=torch.float).to(device)
batch_size=5

N_TRAIN_STEPS = 1000


class classify(nn.Module):
    def __init__(self):
        # initialize as nn.Module
        super(classify, self).__init__()

        
        self.conv1 = nn.Conv2d(1,8,(2,6),stride=2)
        self.conv2 = nn.Conv2d(8,16,(2,6),stride=2)
        self.conv3 = nn.Conv2d(16,32,(2,6),stride=2)
        self.linear1 = nn.Linear(in_features=3360,out_features=5)
        self.linear2 = nn.Linear(in_features=5, out_features=3360)
        self.conv4 = nn.ConvTranspose2d(32,16,(2,6),stride=2)
        self.conv5 = nn.ConvTranspose2d(16,8,(2,8),stride=2)
        self.conv6 = nn.ConvTranspose2d(8,1,(2,6),stride=2)
        self.activation_out = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.activation_out(self.conv1(x))
        x = self.activation_out(self.conv2(x))
        x = self.activation_out(self.conv3(x))
        x = torch.reshape(x,[batch_size,3360])
        x = self.activation_out(self.linear1(x))
        x = self.activation_out(self.linear2(x))
        x = torch.reshape(x,[batch_size,32,5,21])
        x = self.activation_out(self.conv4(x))
        x = self.activation_out(self.conv5(x))
        x = self.activation_out(self.conv6(x))
        output1 = torch.reshape(x,[batch_size,1,NUM_VELOCITY,NUM_SPACE])
        return output1


if __name__ == '__main__':

    classify = classify().to(device)
    
    optimizer = Adam(params=classify.parameters(), lr=0.001)

    loss_crit = nn.L1Loss()
    train_losses = []


    for step in range(N_TRAIN_STEPS):
        
        for p in range(5):

            train_in = train_list[p]

            cl_out = classify(train_in)

            train_loss = loss_crit(cl_out, train_in)
            
            train_losses.append(train_loss.item())

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

        print('Epoch :',step, 'train_loss:',train_loss,':)')

        if train_loss <= 0.0008:
                break


plt.semilogy(np.arange(5*step+5), train_losses, label='Training loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()




samples = tensor(f2, dtype=torch.float).to(device)
batch_size=25

predict = classify(samples).to(device)

samples = samples.to('cpu')
predict = predict.to('cpu')
samples = samples.squeeze()
predict = predict.squeeze()

samples  = samples.detach().numpy()

predict = predict.detach().numpy()


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
