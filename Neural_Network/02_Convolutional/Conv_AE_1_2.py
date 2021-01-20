'''
Convolutional Autoencoder v1.0
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torch.tensor as tensor
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import DataLoader

def progressBar(value, endvalue, train_loss, test_loss, bar_length=20, ):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        train_loss = train_loss
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        #sys.stdout.write("\rEpoch {0}, Train Loss: {2.10f}, Test Loss: {3.10f}")
        sys.stdout.flush()

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.is_available())
code_list = (64,32,16,8,5,2,1)
for i in code_list:

    N_EPOCHS = 6000
    BATCH_SIZE = 4
    lr = 1e-4


    device = 'cpu'

    #load data
    f = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Preprocessing/preprocessed_samples_conv.npy')


    #shuffe or not to shuffle ----> k-means???
    #np.random.shuffle(f)
    f = tensor(f, dtype=torch.float).to(device)

    train_in = f[0:32]
    val_in = f[32:40]

    # plt.imshow(val_in[4].squeeze())
    # plt.show()
    # fig, axs = plt.subplots(nrows=4,ncols=2)
    # axs[0,0].imshow(val_in[0].squeeze())
    # axs[0,1].imshow(val_in[1].squeeze())
    # axs[1,0].imshow(val_in[2].squeeze())
    # axs[1,1].imshow(val_in[3].squeeze())
    # axs[2,0].imshow(val_in[4].squeeze())
    # axs[2,1].imshow(val_in[5].squeeze())
    # axs[3,0].imshow(val_in[6].squeeze())
    # axs[3,1].imshow(val_in[7].squeeze())
    # plt.show()




    train_iterator = DataLoader(train_in, batch_size = BATCH_SIZE)
    test_iterator = DataLoader(val_in, batch_size = int(len(f)*0.2))



    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.convE1 = nn.Conv2d(1,8,(5,10),stride=(4,5))
            self.convE2 = nn.Conv2d(8,16,(4,4),stride=(2,5))
            self.linearE1 = nn.Linear(in_features=256,out_features=i)
            self.act = nn.Tanh()
            #self.act_c = nn.Tanh()

        def forward(self, x):
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            #x = self.act_c(self.linearE1(x))
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=i, out_features=256)
            self.convD1 = nn.ConvTranspose2d(16,8,(4,4),stride=(2,5))
            self.convD2 = nn.ConvTranspose2d(8,1,(5,10),stride=(4,5))
            self.act = nn.Tanh()
            #self.act_c = nn.Tanh()

        def forward(self, x):
            x = self.linearD1(x)
            #x = self.act_c(self.linearD1(x))
            dim = x.shape[0]
            x = torch.reshape(x,[dim,16,2,8])
            x = self.act(self.convD1(x))
            x = self.act(self.convD2(x))
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


    #encoder
    encoder = Encoder()

    #decoder
    decoder = Decoder()

    #Autoencoder
    model = Autoencoder(encoder, decoder).to(device)

    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer,milestones=[3000,5000],verbose=False)

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

                #scheduler.step(loss)

            return test_loss


    train_losses = []
    test_losses = []

    #checkpoint Load
    # checkpoint = torch.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Conv_Nets/Conv_State_Dicts/Conv_AE_STATE_DICT_1_0_1c_2_1.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

        progressBar(epoch,N_EPOCHS, train_loss, test_loss)

        


        # if epoch % 1000 == 0:

        #     #i = randint(0,7)
        #     #i=3
        #     x = val_in[0]
        #     x = x.unsqueeze(0)
        #     predicted = model(x)
        #     x = x.to('cpu')
        #     predicted = predicted.to('cpu')
        #     data = x.detach().numpy()
        #     predict = predicted.detach().numpy()
            
        #     fig, axs = plt.subplots(nrows=2)
        #     org = axs[0].imshow(data.squeeze(),vmin=0, vmax=np.max(data))
        #     pred = axs[1].imshow(predict.squeeze(),vmin=0,vmax=np.max(data))
        #     fig.colorbar(org, ax = axs[0])
        #     fig.colorbar(pred, ax = axs[1])
        #     plt.show()


    # plt.figure()
    # plt.semilogy(np.arange(N_EPOCHS), train_losses, label='Training loss')
    # plt.semilogy(np.arange(N_EPOCHS), test_losses, label='Test loss')
    # plt.legend(loc='upper right')
    # plt.xlabel('trainstep')
    # plt.ylabel('loss')
    # plt.show()


    


    #save the models state dictionary for inference
    torch.save({
        'epoch': epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_losses':train_losses,
        'test_losses': test_losses
        },f'/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Conv_Nets/Code/1_2/CoAE_SD_1_%s.pt'%i)
        
    print(f'FINISHED Training DUDE %s'%i)



