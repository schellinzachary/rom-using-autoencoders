'''
Linear Autoencoder v0.9
'''



import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_EPOCHS = 200
BATCH_SIZE = 32
INPUT_DIM = 40
LATENT_DIM = 5
lr = 1e-3
lam = 1e-4



#load data
f = np.load('preprocessed_samples_lin.npy')
np.random.shuffle(f)
f = tensor(f, dtype=torch.float).to(device)

train_in = f[0:3999]
val_in = f[4000:4999]


train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
test_iterator = DataLoader(val_in)

class Encoder(nn.Module):
    def __init__(self, input_dim, lat_dim):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(in_features=input_dim, 
                                    out_features=lat_dim, bias=False)
        self.activation_out = nn.LeakyReLU()
    def forward(self, x):
        x = self.activation_out(self.linear1(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, lat_dim):
        super(Decoder, self).__init__()
        self.linear2 = nn.Linear(in_features=lat_dim, 
                                out_features=input_dim,bias=False)
        self.activation_out = nn.LeakyReLU()

    def forward(self,x):

        x = self.activation_out(self.linear2(x))
      
        return x


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
encoder = Encoder(INPUT_DIM,LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM,LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

#model.load_state_dict(torch.load('Lin_AE_STATE_DICT_0_9_L5.pt'))
   
optimizer = Adam(params=model.parameters(), lr=lr)

loss_crit = nn.MSELoss()
train_losses = []
val_losses = []



def loss_function(W, x, predicted, h, lam):
    mse = loss_crit(predicted, x)

    dh = h * (1 - h) 

    w_sum = torch.sum(W**2, dim=1)

    w_sum = w_sum.unsqueeze(1)

    contractive_loss = torch.sum(torch.mm(dh**2,w_sum), 0)

    return mse + contractive_loss * lam


def train():

    model.train()

    train_loss = 0.0

    for batch_ndx, x in enumerate(train_iterator):

        x = x.to(device)

        optimizer.zero_grad()

        predicted, h = model(x)

        W = encoder.state_dict()['linear1.weight']


        loss = loss_function(W, x, predicted, h, lam)

        loss.backward()
        train_loss += loss.item()

        optimizer.step()

    return train_loss, x, predicted

def test():

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for i, x in enumerate(test_iterator):

            x = x.to(device)

            predicted, h = model(x)

            W = encoder.state_dict()['linear1.weight']

            loss = loss_function(W, x, predicted, h, lam)
            test_loss += loss.item()

        return test_loss



        
        print('Epoch :',step, 'train_loss:',train_loss,':)')

test_losses = []
val_losses = []

for n_iter in range(N_EPOCHS):

    train_loss, x, predicted = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {n_iter}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')


    if n_iter % 100 == 0:

        x = x.to('cpu')
        predicted = predicted.to('cpu')
        data = x.detach().numpy()
        predict = predicted.detach().numpy()

        plt.plot(x[-1], label='Original')
        plt.plot(predict[-1], label='Predicted')
        plt.legend()
        plt.show()

plt.figure()
plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()


np.save('Train_Loss_CAE_0_1_L5.npy',train_losses)
np.save('Test_Loss_CAE_0_1_L5.npy',test_losses)




#save the models state dictionary for inference
torch.save(model.state_dict(),'CAE_STATE_DICT_0_1_L5.pt')