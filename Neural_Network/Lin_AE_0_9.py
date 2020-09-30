'''
Linear Autoencoder v0.9
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

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

N_EPOCHS = 400
BATCH_SIZE = 16
INPUT_DIM = 40
LATENT_DIM = 5
lr = 1e-3



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
                                    out_features=lat_dim)
        self.activation_out = nn.LeakyReLU()
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
        a =  enc.linear1.weight
        dec.linear2.weight = nn.Parameter(torch.transpose(a,0,1))

    def forward(self, x):
        z = self.enc(x)
        predicted = self.dec(z)
        return predicted

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#encoder
encoder = Encoder(INPUT_DIM,LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM,LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

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

        loss = loss_crit(predicted,x)

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

            loss = loss_crit(predicted,x)
            test_loss += loss.item()

        return test_loss

test_losses = []
val_losses = []

#checkpoint Load
checkpoint = torch.load('Lin_AE_STATE_DICT_0_9_L5_substr50_test.pt')
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_o = checkpoint['epoch']
train_loss = checkpoint['train_loss']
test_loss = checkpoint['test_loss']
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']


for epoch in range(N_EPOCHS):

    train_loss = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}')


    # if epoch % 100 == 0:

    #     i = randint(0,999)
    #     x = val_in[i].to(device)

    #     predicted = model(x)
    #     x = x.to('cpu')
    #     predicted = predicted.to('cpu')
    #     data = x.detach().numpy()
    #     predict = predicted.detach().numpy()
        
    #     plt.plot(x, label='Original')
    #     plt.plot(predict, label='Predicted')
    #     plt.legend()
    #     plt.show()

plt.figure()
plt.semilogy(np.arange(N_EPOCHS+600), train_losses, label='Training loss')
plt.semilogy(np.arange(N_EPOCHS+600), test_losses, label='Test loss')
plt.legend(loc='upper right')
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()


np.save('Train_Loss_Lin_0_9_L5_substr50_test.npy',train_losses)
np.save('Test_Loss_Lin_0_9_L5_substr50_test.npy',test_losses)




#save the models state dictionary for inference
torch.save({
    'epoch': epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'test_loss': test_loss,
    'train_losses':train_losses,
    'test_losses': test_losses
    },'Lin_AE_STATE_DICT_0_9_L5_substr50_test.pt')