'''
Convolutional Autoencoder v1.0
'''

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta, Adagrad
import torch.tensor as tensor
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import DataLoader

parent_dir = "/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Conv_Nets/Learning_rate"

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.is_available())



N_EPOCHS = 6000
BATCH_SIZE = 4
lr = 1e-4


device = 'cpu'

#load data
f = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/preprocessed_samples_conv.npy')

#shuffe or not to shuffle ----> k-means???
#np.random.shuffle(f)
f = tensor(f, dtype=torch.float).to(device)

train_in = f[0:31]
val_in = f[32:39]

print(val_in[1].shape)


train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))
print('test_it',len(test_iterator),'train_it',len(train_iterator))


N_TRAIN_STEPS = 2000



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convE1 = nn.Conv2d(1,8,(3,2),stride=2)
        self.convE2 = nn.Conv2d(8,16,(2,2),stride=2)
        self.convE3 = nn.Conv2d(16,32,(2,2),stride=2)
        self.convE4 = nn.Conv2d(32,64,(3,3),stride=(1,2))
        self.linearE1 = nn.Linear(in_features=768,out_features=3)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.convE1(x))
        x = self.act(self.convE2(x))
        x = self.act(self.convE3(x))
        x = self.act(self.convE4(x))
        original_size = x.size()
        x = x.view(original_size[0],-1)
        x = self.linearE1(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linearD1 = nn.Linear(in_features=3, out_features=768)
        self.convD1 = nn.ConvTranspose2d(64,32,(3,3),stride=(1,2))
        self.convD2 = nn.ConvTranspose2d(32,16,(2,2),stride=2)
        self.convD3 = nn.ConvTranspose2d(16,8,(2,2),stride=2)
        self.convD4 = nn.ConvTranspose2d(8,1,(3,2),stride=2)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.linearD1(x)
        dim = x.shape[0]
        x = torch.reshape(x,[dim,64,1,12])
        x = self.act(self.convD1(x))
        x = self.act(self.convD2(x))
        x = self.act(self.convD3(x))
        x = self.act(self.convD4(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, enc, dec):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#encoder
encoder = Encoder()

#decoder
decoder = Decoder()

#Autoencoder
model = Autoencoder(encoder, decoder).to(device)

optimizer = Adam(params=model.parameters(), lr=lr)

loss_crit = nn.MSELoss(reduction='mean')
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

train_losses = []
test_losses = []

#checkpoint Load
# checkpoint = torch.load('Lin_AE_STATE_DICT_0_9_L5_substr50_test.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch_o = checkpoint['epoch']
# train_loss = checkpoint['train_loss']
# test_loss = checkpoint['test_loss']
# train_losses = checkpoint['train_losses']
# test_losses = checkpoint['test_losses']


for epoch in range(N_EPOCHS):

    train_loss = train()
    test_loss = test()

    #save and print the loss
    train_loss /= len(train_iterator)
    test_loss /= len(test_iterator)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch}, Train Loss: {train_loss:.10f}, Test Loss: {test_loss:.10f}')


    if epoch % 100 == 0:

        #i = randint(0,7)
        #i=3
        x = val_in[0]
        x = x.unsqueeze(0)
        predicted = model(x)
        x = x.to('cpu')
        predicted = predicted.to('cpu')
        data = x.detach().numpy()
        predict = predicted.detach().numpy()
        
        #plt.plot(x, label='Original')
        plt.imshow(predict.squeeze(), label='Predicted')
        plt.legend()
        plt.show()

    if test_loss <= 9.9e-5:
        break

# plt.figure()
# plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
# plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
# plt.legend(loc='upper right')
# plt.xlabel('trainstep')
# plt.ylabel('loss')
# plt.show()


# np.save('Train_Loss_Lin_0_9_L5_substr50_test.npy',train_losses)
# np.save('Test_Loss_Lin_0_9_L5_substr50_test.npy',test_losses)




#save the models state dictionary for inference
torch.save({
    'epoch': epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'test_loss': test_loss,
    'train_losses':train_losses,
    'test_losses': test_losses
    },'/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Conv_Nets/Conv_State_Dicts/Conv_AE_STATE_DICT_0_9_L3_B4_lr-4_Tanh.pt')


