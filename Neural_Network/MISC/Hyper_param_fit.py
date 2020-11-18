'''
Random search for hyperparameters
'''

import umpy as np
from sklearn import model linear_models
from sklearn.model_selection import RandomizedSearchCV

#Hyperparameters
log_learning_rate = np.random.randint(-7,0)
hidden_layers = np.random.randint(1,5)
batch_size = np.random.randint(2,33)
input_activation = [nn.Sigmoid,nn.Tanh,nn.ReLU,nn.ELU,nn.PreLU]
hidden_activation = [nn.Sigmoid,nn.Tanh,nn.ReLU,nn.ELU,nn.PreLU]
output_activation = [nn.Sigmoid,nn.Tanh,nn.ReLU,nn.ELU,nn.PreLU]
in_features = [20,10,5]
out_features = [20,10,5]

hyperparameters = [log_learning_rate, hidden_units, batch_size, hidden_activation, input_activation, output_activation,in_features]

#Estimtor architecture

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lat_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, 
                                    out_features=hidden_dim)
        for l in range(hidden_layers + 1):
            self.add_module('layer_' + str(l), torch.nn.Linear(in_features=in_features,out_features=out_features))
            self.add_module('activ_' + str(l), input_activation)


    def forward(self, x):

    	x = self.activation_out(self.linear1(x))
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
