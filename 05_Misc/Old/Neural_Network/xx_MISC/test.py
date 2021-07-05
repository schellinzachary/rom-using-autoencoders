import numpy as np
import torch
import torch.nn as nn
import torch.tensor as tensor
import matplotlib.pyplot as plt

# All tensors are loaded in standard format of tensor shape : (40x5000) as used in POD

org = np.load('/home/fusilly/ROM_using_Autoencoders/data_sod/original_data_in_format.npy')

rec = np.load('/home/fusilly/ROM_using_Autoencoders/data_sod/SVD_reconstruction.npy')


print(org.shape)


def dense(x):

    dense = np.zeros([25,200])
    n=0

    for k in range(25):
        for i in range(200):
            dense[k,i] = np.sum(x[:,i+n]) * 0.5128
   
        n += 200
    return dense


def net(c):

	INPUT_DIM = 40
	LATENT_DIM = 5


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

	    def forward(self, x):
	        z = self.enc(x)
	        predicted = self.dec(z)
	        return predicted, z

	class Swish(nn.Module):
	    def forward(self, x):
	        return x * torch.sigmoid(x)

	#encoder
	encoder = Encoder(INPUT_DIM,LATENT_DIM)

	#decoder
	decoder = Decoder(INPUT_DIM,LATENT_DIM)

	#Autoencoder
	model = Autoencoder(encoder, decoder)


	model.load_state_dict(torch.load('Lin_AE_STATE_DICT_0_9_L5_substr50.pt',map_location='cpu')['model_state_dict'])

	model.eval()

	# load original data
	c = c.T
	#Inference

	c = tensor(c, dtype=torch.float)
	predict, z = model(c)
	c = c.detach().numpy()
	predict = predict.detach().numpy()

	return predict



dense_org = dense(org)
dense_rec = dense(rec)
predi = net(org)
dense_net = dense(predi.T)
print(np.sum (np.abs((org-rec)/5000)))
print(np.sum(np.abs(org - predi.T))/5000)
print(np.sum(np.abs(dense_rec - dense_org))/25)
print(np.sum(np.abs(dense_org - dense_net))/25)


POD = np.sum(np.abs(org - rec),axis=0)
NET = np.sum(np.abs(org - predi.T),axis=0)
plt.plot(POD,'*''k')
plt.plot(NET,'v''b')

plt.show()


