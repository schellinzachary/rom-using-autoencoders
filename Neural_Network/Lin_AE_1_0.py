'''
Linear Autoencoder v1.0
'''



import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io as sio
from random import randint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_EPOCHS = 600
BATCH_SIZE = 16
INPUT_DIM = 40
HIDDEN_DIM = 20
LATENT_DIM = 5
lr = 1e-3



#load data
f = np.load('preprocessed_samples_lin.npy')
np.random.shuffle(f)
f = tensor(f, dtype=torch.float).to(device)

train_in = f[0:3999]
val_in = f[4000:4999]


train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(in_features=input_dim, 
                                    out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, 
                                    out_features=lat_dim)
        self.activation_out = nn.LeakyReLU()
        self.activation_out1 = nn.Tanh()
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
        z = self.enc(x)
        predicted = self.dec(z)
        return predicted

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
#encoder
encoder = Encoder(INPUT_DIM,HIDDEN_DIM, LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

#model.load_state_dict(torch.load('Lin_AE_STATE_DICT_1_0_L5_sigmoid.pt'))
   
optimizer = Adam(params=model.parameters(), lr=lr)

loss_crit = nn.MSELoss()
train_losses = []
val_losses = []


def train():

    model.train()

    train_loss = 0.0

    for batch_ndx, x in enumerate(train_iterator):

        x = x.to(device)

        optimizer.zero_grad()

        predicted = model(x)

        loss = loss_crit(x,predicted)

        loss.backward()
        train_loss += loss.item()

        optimizer.step()

    return train_loss

def test():

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for i, x in enumerate(test_iterator):

            x = x.to(device)

            predicted = model(x)

            loss = loss_crit(x,predicted)
            test_loss += loss.item()

        return test_loss, x

test_losses = []
val_losses = []

for n_iter in range(N_EPOCHS):

    train_loss = train()
    test_loss, x = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {n_iter}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}')


    # if n_iter % 300 == 0:

    #      i = randint(0,999)
    #      x = val_in[i].to(device)

    #      predicted = model(x)
    #      x = x.to('cpu')
    #      predicted = predicted.to('cpu')
    #      data = x.detach().numpy()
    #      predict = predicted.detach().numpy()
        
    #      plt.plot(x, label='Original')
    #      plt.plot(predict, label='Predicted')
    #      plt.legend()
    #      plt.show()

plt.figure()
plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()


np.save('Train_Loss_Lin_1_0_L5.npy',train_losses)
np.save('Test_Loss_Lin_1_0_L5.npy',test_losses)




#save the models state dictionary for inference
torch.save(model.state_dict(),'Lin_AE_STATE_DICT_1_0_L5_16_lr-3_TH.pt')