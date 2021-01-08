'''
Linear Autoencoder v1.1
'''



import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_EPOCHS = 300
BATCH_SIZE = 16
INPUT_DIM = 40
HIDDEN_DIM = 20
HIDDEN_DIM2 = 10
LATENT_DIM = 3
lr = 1e-3



#load data
f = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Preprocessing/preprocessed_samples_lin.npy')
np.random.shuffle(f)
f = tensor(f, dtype=torch.float).to(device)

train_in = f[0:4000]
val_in = f[4000:5000]


train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(in_features=input_dim, 
                                    out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, 
                                    out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, 
                                    out_features=lat_dim)
        self.activation_out = nn.Tanh()
        self.activation_out1 = nn.Tanh()
    def forward(self, x):
        x = self.activation_out(self.linear1(x))
        x = self.activation_out(self.linear2(x))
        x = self.activation_out1(self.linear3(x))

        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Decoder, self).__init__()

        self.linear4 = nn.Linear(in_features=lat_dim, 
                                    out_features=hidden_dim)
        self.linear5 = nn.Linear(in_features=hidden_dim, 
                                    out_features=hidden_dim)
        self.linear6 = nn.Linear(in_features=hidden_dim, 
                                    out_features=input_dim)
        self.activation_out = nn.Tanh()

    def forward(self,x):
        x = self.activation_out(self.linear4(x))
        x = self.activation_out(self.linear5(x))
        x = self.activation_out(self.linear6(x))
      
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
encoder = Encoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#decoder
decoder = Decoder(INPUT_DIM,HIDDEN_DIM,LATENT_DIM)

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)
    
optimizer = Adam(params=model.parameters(), lr=lr)

loss_crit = nn.MSELoss()
train_losses = []
val_losses = []

#model.load_state_dict(torch.load('Lin_AE_STATE_DICT_1_1_32_L3_lr-7_LR.pt'))

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

    return train_loss, x, predicted

def test():

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for i, x in enumerate(test_iterator):

            x = x.to(device)

            predicted = model(x)

            loss = loss_crit(x,predicted)
            test_loss += loss.item()

        return test_loss



        
        print('Epoch :',step, 'train_loss:',train_loss,':)')

test_losses = []
val_losses = []

for epoch in range(N_EPOCHS):

    train_loss, x, predicted = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}')


    # if n_iter % 1000 == 0:

    #     x = x.to('cpu')
    #     predicted = predicted.to('cpu')
    #     data = x.detach().numpy()
    #     predict = predicted.detach().numpy()

    #     plt.plot(x[-1], label='Original')
    #     plt.plot(predict[-1], label='Predicted')
    #     plt.legend()
    #     plt.show()

plt.figure()
plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
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
    },f'/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/1_Lin_AE_Nets/LIN_AE_State_Dicts/1_1/AE_SD_1.pt')
    
print(f'FINISHED Training DUDE')
