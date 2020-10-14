'''
Linear Autoencoder v0.9
'''



import numpy as np
import torch
import torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'


N_EPOCHS = 1000
BATCH_SIZE = 16
INPUT_DIM = 40
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
    def __init__(self, input_dim, lat_dim):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(in_features=input_dim, 
                                    out_features=lat_dim, bias=False)
        self.activation_out = nn.Tanh()
    def forward(self, x):
        x = self.activation_out(self.linear1(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, lat_dim):
        super(Decoder, self).__init__()
        self.linear2 = nn.Linear(in_features=lat_dim, 
                                out_features=input_dim)
        self.activation_out = nn.LeakyReLU()

    def forward(self,x):
        x = self.activation_out(self.linear2(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
        # #tie the weights
        # a =  enc.linear1.weight
        # dec.linear2.weight = nn.Parameter(torch.transpose(a,0,1))
        
    def forward(self, x):
        h = self.enc(x)
        predicted = self.dec(h)
        #print(torch.sum(torch.abs(self.dec.linear2.weight - torch.transpose(self.enc.linear1.weight,0,1))))
        return predicted, h

# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)

#encoder
encoder = Encoder(INPUT_DIM,LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM,LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

#model.load_state_dict(torch.load('CAE_STATE_DICT_0_1_L5_16_TH_subtr50.pt')['model_state_dict'])
   
optimizer = Adam(params=model.parameters(), lr=lr)

loss_crit = nn.MSELoss()


def loss_function(W, x, predicted, h, lam):
    mse = loss_crit(predicted, x)

    #     #Sigmoid
    # dh = h * (1 - h)

    # w_sum = torch.sum(W**2,dim=1)

    # w_sum = w_sum.unsqueeze(1)

    # contractive_loss = torch.sum(torch.mm(dh**2, w_sum))

        #LeakyReLU

    # dh = torch.where(W >= 0 , torch.ones(1), torch.ones(1)*1e-2) 

    # j = dh*W

    # contractive_loss  = torch.sqrt(torch.sum(j**2))

        #Tanh
    dh = 1 - h**2

    w_sum = torch.sum(W**2,dim=1)

    w_sum = w_sum.unsqueeze(1)

    contractive_loss = torch.sum(torch.mm(dh**2, w_sum))

    return mse + contractive_loss * lam, mse, contractive_loss


def train():

    model.train()

    train_loss = 0.0
    train_mse_loss = 0.0
    train_con_loss = 0.0

    for batch_ndx, x in enumerate(train_iterator):

        x = x.to(device)

        optimizer.zero_grad()

        predicted, h = model(x)

        W = encoder.state_dict()['linear1.weight']

        loss, mse, con = loss_function(W, x, predicted, h, lam)

        loss.backward()
        train_loss += loss.item()
        train_mse_loss += mse.item()
        train_con_loss += con.item()

        optimizer.step()

    return train_loss, train_mse_loss, train_con_loss

def test():

    model.eval()

    test_loss = 0.0
    test_mse_loss = 0.0
    test_con_loss = 0.0

    with torch.no_grad():
        for i, x in enumerate(test_iterator):

            x = x.to(device)

            predicted, h = model(x)

            W = encoder.state_dict()['linear1.weight']

            loss, mse, con = loss_function(W, x, predicted, h, lam)
            test_loss += loss.item()
            test_mse_loss += mse.item()
            test_con_loss += con.item()

    return test_loss, test_mse_loss, test_con_loss



test_losses = []
test_mse_losses = []
test_con_losses = []
train_losses = []
train_mse_losses =[]
train_con_losses = []



for epoch in range(N_EPOCHS):

    train_loss, train_mse_loss, train_con_loss = train()
    test_loss, test_mse_loss, test_con_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    train_mse_loss /= len(train_iterator)
    train_con_loss /= len(train_iterator)


    
    train_mse_losses.append(train_mse_loss)
    train_con_losses.append(train_con_loss)
    train_losses.append(train_loss)

    test_mse_losses.append(test_mse_loss)
    test_con_losses.append(test_con_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f},Test CON Loss: {test_con_loss:.5f}, Train CON Loss: {train_con_loss:.5f}')


    # if epoch % 100 == 0:

    #     x = x.to('cpu')
    #     predicted = predicted.to('cpu')
    #     data = x.detach().numpy()
    #     predict = predicted.detach().numpy()

    #     plt.plot(x[-1], label='Original')
    #     plt.plot(predict[-1], label='Predicted')
    #     plt.legend()
    #     plt.show()

plt.figure()
plt.semilogy(np.arange(N_EPOCHS), train_con_losses, label='Training loss')
plt.semilogy(np.arange(N_EPOCHS), test_con_losses, label='Test loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()







#save the models state dictionary for inference
torch.save({
    'epoch': epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'test_loss': test_loss,
    'train_losses':train_losses,
    'test_losses': test_losses
    },'CAE_STATE_DICT_0_1_L5_16_TH_subtr50.pt')