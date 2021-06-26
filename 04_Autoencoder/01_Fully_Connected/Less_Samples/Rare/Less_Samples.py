'''
Intrinsic Variables Hydro
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/02_data_sod/sod25Kn0p01/f.mat"


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def twoD(f):
  x=f.shape[2]
  t=f.shape[0]
  v=f.shape[1]
  #Submatrix
  c = np.empty((t*x,v))
  n = 0

  #Build 2D-Version
  for i in range(t):                                         
    for j in range(x):
      c[j+n,:]=f[i,:,j]
    n +=200
  return(c)

#set variables
del_var = [2,3,4,5]

def save_checkpoint(k_models):
    k_models = np.array(k_models)
    k_models = k_models[k_models[:,0].argsort()]

    return np.ndarray.tolist(k_models[:3])


class Encoder(nn.Module):
    def __init__(self,int_var=None):
        super(Encoder, self).__init__()
        self.int_var = int_var
        self.add_module('layer_1', torch.nn.Linear(in_features=40,out_features=40))
        self.add_module('activ_1', nn.ReLU())
        self.add_module('layer_c',nn.Linear(in_features=40, out_features=self.int_var))
        self.add_module('activ_c', nn.ReLU())
    def forward(self, x):
        for _, method in self.named_children():
            x = method(x)
        return x

class Decoder(nn.Module):
    def __init__(self,int_var=None):
        super(Decoder, self).__init__()
        self.int_var = int_var
        self.add_module('layer_c',nn.Linear(in_features=self.int_var, out_features=40))
        self.add_module('activ_c', nn.ReLU())
        self.add_module('layer_4', nn.Linear(in_features=40,out_features=40))
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

N_EPOCHS = 5000

for idx, del_var in enumerate(del_var):

    #load data
    f = sio.loadmat(join(home,loc_data)) 
    f  = f['f']

    f = f[::del_var,:,:]
    f = twoD(f)
    f = tensor(f, dtype=torch.float).to(device)

    len_t = int(len(f)*.8)
    len_v = int(len(f)*.2)
    train_in, val_in = random_split(f,[len_t,len_v])

    train_dataset = DataLoader(train_in,
                                batch_size=4
                                )
    val_dataset = DataLoader(val_in, 
        batch_size = int(len(f)*0.2)
        ) 

    
    #encoder
    encoder = Encoder(int_var=5)

    #decoder
    decoder = Decoder(int_var=5)

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)
    print(model)

    optimizer = Adam(params=model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=3000, gamma=0.1)

    loss_crit = nn.MSELoss()
    train_losses = []
    val_losses = []


    def train():

        model.train()

        train_loss = 0.0

        for batch_ndx, x in enumerate(train_dataset):

            x = x.to(device)

            optimizer.zero_grad()

            predicted = model(x)

            loss = loss_crit(predicted,x)

            loss.backward()
            train_loss += loss.item()

            optimizer.step()

        return train_loss

    def test():

        model.eval()

        test_loss = 0

        with torch.no_grad():
            for i, x in enumerate(val_dataset):

                x = x.to(device)

                predicted = model(x)

                loss = loss_crit(predicted,x)
                test_loss += loss.item()

            return test_loss

    test_losses = []
    val_losses = []
    k_models = []


    for epoch in tqdm(range(N_EPOCHS)):
        train_loss = train()
        test_loss = test()
        scheduler.step()

        #save and print the loss
        train_loss /= len(train_dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        
        k_models.append((
            test_loss,
            epoch,
            model.state_dict())
        )
        if (epoch > 1) and(epoch % 10 == 0):
            k_models = save_checkpoint(k_models)
        #if epoch % 1000 == 0:
            #print("Top 3 Models are from epochs %s" %k_models[:3,1]) 
        
    #save top 3 models
    for i in range(3):
        k_models = np.array(k_models)
        torch.save({
    'epoch': k_models[i,1],
    'model_state_dict':k_models[i,2],
    },'Results/del_var-{}-epoch{}-val_loss{:.3E}.pt'.format(del_var,
        k_models[i,1],
        k_models[i,0]))

    #save last model
    torch.save({
        'epoch': epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses':train_losses,
        'test_losses': test_losses
        },'Results/last-{}.pt'.format(del_var))
