'''
Parameterstudy_01_Layer_Size
'''

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
import scipy.io as sio
import sys

def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class params():
    N_EPOCHS = 2000
    BATCH_SIZE = 16
    INPUT_DIM = 40
    H_SIZES = 40
    LATENT_DIM = 5
    lr = 1e-4

class data():
    #load data
    f = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_2D.npy')
    unsh = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy')
    f = tensor(f, dtype=torch.float).to(device)
    unsh = tensor(unsh,dtype=torch.float).to(device)
    

    train_in = f[0:3999]
    val_in = f[4000:4999]
    train_iterator = DataLoader(train_in, batch_size = params.BATCH_SIZE)
    test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.add_module('layer_1', torch.nn.Linear(in_features=params.INPUT_DIM,out_features=params.H_SIZES))
        self.add_module('activ_1', nn.Tanh())
        self.add_module('layer_c',nn.Linear(in_features=params.H_SIZES, out_features=params.LATENT_DIM))
        self.add_module('activ_c', nn.Tanh())
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.add_module('layer_c',nn.Linear(in_features=params.LATENT_DIM, out_features=params.H_SIZES))
        self.add_module('activ_c', nn.Tanh())
        self.add_module('layer_4', nn.Linear(in_features=params.H_SIZES,out_features=params.INPUT_DIM))
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x



class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z = self.enc(x)
        predicted = self.dec(z)
        return predicted

#encoder
encoder = Encoder()

#decoder
decoder = Decoder()

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)
print(model)

optimizer = Adam(params=model.parameters(), lr=params.lr)



def shapeback_code(z):
    c = torch.empty((25,params.LATENT_DIM,200))
    n=0
    for i in range(25):
        for p in range(200):
          c[i,:,p] = z[p+n,:]
        n += 200
    return(c) # shaping back the code



def cons_loss(unsh):
    x = encoder(unsh)
    x = shapeback_code(x)

    print('g')
    return(torch.sum(x))




loss_crit = nn.MSELoss()
train_losses = []
val_losses = []


def train():

    model.train()

    train_loss = 0.0

    for batch_ndx, x in enumerate(data.train_iterator):

        x = x.to(device)

        optimizer.zero_grad()

        predicted = model(x)

        loss_mse = loss_crit(predicted,x)
        loss_cons = cons_loss(data.unsh)


        loss = loss_mse + loss_cons

        loss.backward()
        train_loss += loss.item()

        optimizer.step()

    return train_loss
def test():

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for i, x in enumerate(data.test_iterator):

            x = x.to(device)

            predicted = model(x)

            loss = loss_crit(predicted,x)
            test_loss += loss.item()

        return test_loss

test_losses = []
val_losses = []

#checkpoint Load
checkpoint = torch.load('Results/LeakyReLU_Tanh_test-4000.pt')
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])
epoch_o = checkpoint['epoch']
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']


for epoch in range(params.N_EPOCHS):

    train_loss = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(data.train_iterator)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    progressBar(epoch,params.N_EPOCHS)
# save the models state dictionary for inference
torch.save({
    'epoch': epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses':train_losses,
    'test_losses': test_losses
    },'Results/code.pt')