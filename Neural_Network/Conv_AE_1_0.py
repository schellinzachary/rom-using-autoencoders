'''
Convolutional Autoencoder v1.0
'''


import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta, Adagrad
import torch.tensor as tensor
import matplotlib.pyplot as plt
import scipy.io as sio

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.is_available())
data_set1 = sio.loadmat('/home/zachary/tubCloud/BA/data_sod/sod25Kn0p00001/f.mat')
data_set2 = sio.loadmat('/home/zachary/tubCloud/BA/data_sod/sod25Kn0p01/f.mat')

data_set1 = data_set1['f']
data_set2  = data_set2['f']


dataset = np.concatenate((data_set1,data_set2),axis=0)
np.random.shuffle(dataset)
dataset = np.expand_dims(dataset, axis=1)
train_list = dataset

#val_list = dataset[36:-1]

#np.split(train_list,5,axis=0)


train_list = tensor(train_list,dtype=torch.float).to(device)
#val_list = tensor(val_list,dtype=torch.float).to(device)


N_TRAIN_STEPS = 20000



class Encoder(nn.Module):
    def __init__(self):
        # initialize as nn.Module
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,8,(2,6),stride=2)
        self.conv2 = nn.Conv2d(8,16,(2,6),stride=2)
        self.conv3 = nn.Conv2d(16,32,(2,6),stride=2)
        self.linear1 = nn.Linear(in_features=3360,out_features=5)
        self.activation_out = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.activation_out(self.conv1(x))
        x = self.activation_out(self.conv2(x))
        x = self.activation_out(self.conv3(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.activation_out(self.linear1(x))
        return x


class Decoder(nn.Module):

    def __init__(self):

        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(in_features=5, out_features=3360)
        self.conv4 = nn.ConvTranspose2d(32,16,(2,6),stride=2)
        self.conv5 = nn.ConvTranspose2d(16,8,(2,8),stride=2)
        self.conv6 = nn.ConvTranspose2d(8,1,(2,6),stride=2)
        self.activation_out = Swish()

    def forward(self, x):

        x = self.activation_out(self.linear2(x))
        dim = x.shape[0]
        x = torch.reshape(x,[dim,32,5,21])
        x = self.activation_out(self.conv4(x))
        x = self.activation_out(self.conv5(x))
        x = self.activation_out(self.conv6(x))
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

if __name__ == '__main__':

    model = AutoEncoder().to(device)
    
    optimizer = Adam(params=model.parameters(), lr=0.001)

    loss_crit = nn.L1Loss()
    train_losses = []
    #val_losses = []

    for step in range(N_TRAIN_STEPS):

        train_out = model(train_list)
        #val_out  = model(val_list)

        train_loss = loss_crit(train_out, train_list)
        #val_loss = loss_crit(val_out, val_list)
        
        train_losses.append(train_loss.item())
        #val_losses.append(val_loss.item())

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        print('Epoch :',step, 'train_loss:',train_loss)

        if train_loss <= 0.0005:
                break


    plt.semilogy(np.arange(step+1), train_losses, label='Training loss')
        #plt.semilogy(n.arange(step+1), val_losses, label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('trainstep')
    plt.ylabel('loss')
    plt.show()


data_set2 = np.expand_dims(data_set2, axis=1)
samples = tensor(data_set2, dtype=torch.float).to(device)
model = AutoEncoder().to(device)
predict = model(samples)
samples = samples.to('cpu')
predict = predict.to('cpu')

samples  = samples.detach().numpy()
predict = predict.detach().numpy()

np.save('out_samples',samples)
np.save('out_predict',predict)
