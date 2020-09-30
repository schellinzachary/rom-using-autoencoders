'''
Linear Autoencoder v1.0
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
from random import randint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

N_EPOCHS = 1000
BATCH_SIZE = 16
INPUT_DIM = 40
HIDDEN_DIM = 20
LATENT_DIM = 5
lr = 1e-3
lam = 1e-4



#load data
f = np.load('preprocessed_samples_lin_substract50.npy')
np.random.shuffle(f)
f = tensor(f, dtype=torch.float).to(device)


train_in = f[0:2999]
val_in = f[3000:3749]


train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(in_features=input_dim, 
                                    out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, 
                                    out_features=lat_dim, bias=False)
        self.activation_out = nn.LeakyReLU()
        self.activation_out1 = nn.LeakyReLU()
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
        #tie the weights
        a = enc.linear1.weight
        b = enc.linear2.weight
        dec.linear4.weight = nn.Parameter(torch.transpose(a,0,1))
        dec.linear3.weight = nn.Parameter(torch.transpose(b,0,1))

    def forward(self, x):
        h = self.enc(x)
        predicted = self.dec(h)
        #print(torch.sum(torch.abs(self.dec.linear4.weight - torch.transpose(self.enc.linear1.weight,0,1))))
        return predicted, h

#encoder
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

#model.load_state_dict(torch.load('CAE_STATE_DICT_1_0_L4_16_substr50xyz.pt'))
   
optimizer = Adam(params=encoder.parameters(), lr=lr)

loss_crit = nn.MSELoss()
train_losses = []
val_losses = []



def loss_function(W, x, predicted, h, lam):

    mse = loss_crit(predicted, x)

    dh = torch.where(W >= 0 , torch.ones(1), torch.ones(1)*1e-2) 

    j = dh * W


    contractive_loss  = torch.sqrt(torch.sum(j**2))

    # dh = h * (1 - h)

    # w_sum = torch.sum(W**2,dim=1)

    # w_sum = w_sum.unsqueeze(1)

    # contractive_loss = torch.sum(torch.mm(dh**2, w_sum))

    return mse + contractive_loss * lam, contractive_loss


def train():

    model.train()

    train_loss = 0.0
    con_train_loss = 0
    for batch_ndx, x in enumerate(train_iterator):

        x = x.to(device)

        optimizer.zero_grad()

        predicted, h = model(x)

        W = encoder.state_dict()['linear2.weight']

        loss, c_loss = loss_function(W, x, predicted, h, lam)

        loss.backward()
        train_loss += loss.item()
        con_train_loss += c_loss.item()

        optimizer.step()

    return train_loss, x, predicted, con_train_loss

def test():

    model.eval()

    test_loss = 0
    con_test_loss = 0
    with torch.no_grad():
        for i, x in enumerate(test_iterator):

            x = x.to(device)

            predicted, h = model(x)

            W = encoder.state_dict()['linear2.weight']

            loss, c_loss = loss_function(W, x, predicted, h, lam)

            test_loss += loss.item()
            con_test_loss += c_loss.item()
        return test_loss, con_test_loss


test_losses = []
val_losses = []

for n_iter in range(N_EPOCHS):

    train_loss, x, predicted, con_train_loss = train()
    test_loss, con_test_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    con_train_loss /= len(train_iterator)
    #test_loss /= int(len(f*0.2))


    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {n_iter}, Train Loss: {train_loss:.5f}, con_train: {con_train_loss:.5f}, Test Loss: {test_loss:.5f}, con_test: {con_test_loss:.5f}')


    # if n_iter % 300 == 0:

    #     i = randint(0,999)
    #     x = val_in[i].to(device)

    #     predicted, h = model(x)
    #     x = x.to('cpu')
    #     predicted = predicted.to('cpu')
    #     data = x.detach().numpy()
    #     predict = predicted.detach().numpy()
        
    #     plt.plot(x, label='Original')
    #     plt.plot(predict, label='Predicted')
    #     plt.legend()
    #     plt.show()


plt.figure()
plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()


np.save('Train_Loss_CAE_1_0_L5_16_subtr50_tiedWeights_LR.npy',train_losses)
np.save('Test_Loss_CAE_1_0_L5_substr503_tiedWeights_LR.npy',test_losses)




#save the models state dictionary for inference
torch.save(model.state_dict(),'CAE_STATE_DICT_1_0_L5_16_substr50_tiedWeights_LR.pt')