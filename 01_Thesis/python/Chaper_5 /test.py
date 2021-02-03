import numpy as np
import torch
import torch.nn as nn
import torch.tensor as tensor




device="cpu"



class net():

    class Encoder(nn.Module):
        def __init__(self,level):
            if level == "hy":
                self.level = 3
            else:
                self.level= 5
            super(net.Encoder, self).__init__()
            self.m = nn.ZeroPad2d((0,0,1,1))
            self.convE1 = nn.Conv2d(1,32,(6,10),stride=(3,10))
            self.convE2 = nn.Conv2d(32,64,(4,10),stride=(4,10))
            self.linearE1 = nn.Linear(in_features=256,out_features=self.level)
            self.add_module('act',nn.SiLU())


        def forward(self, x):
            x = self.m(x)
            x = self.act(self.convE1(x))
            x = self.act(self.convE2(x))
            original_size = x.size()
            x = x.view(original_size[0],-1)
            x = self.linearE1(x)
            return x


    class Decoder(nn.Module):
        def __init__(self,level):
            if level == "hy":
                self.level = 3
            else:
                self.level= 5
            print(self.level)
            super(net.Decoder, self).__init__()
            self.linearD1 = nn.Linear(in_features=self.level, out_features=256)
            self.convD1 = nn.ConvTranspose2d(64,32,(4,10),stride=(4,10))
            self.convD2 = nn.ConvTranspose2d(32,1,(4,10),stride=(3,10))
            self.add_module('act',nn.SiLU())


        def forward(self, x):
            x = self.linearD1(x)
            dim = x.shape[0]
            x = torch.reshape(x,[dim,64,2,2])
            x = self.act(self.convD1(x))
            x = self.act(self.convD2(x))
            return x



    class Autoencoder(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, x):
            z = self.enc(x)
            x = self.dec(z)
            return x, z

class modeli(object):
    def __init__(self,level):
        self.level = level
    def load(level,c):
        #Load model best performing model from Parameterstudy
        #####################################################
        encoder = net.Encoder(level)
        decoder = net.Decoder(level)
        model = net.Autoencoder(encoder, decoder).to(device)
        print('Conv_{}'.format(level))
        checkpoint = torch.load('State_Dict/Conv_{}.pt'.format(level))
        model.load_state_dict(checkpoint['model_state_dict'])
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        N_EPOCHS = checkpoint['epoch']
        rec, code = model(c)
        return rec,code