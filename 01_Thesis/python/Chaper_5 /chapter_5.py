### Producing the plots for for Chapter 5
### 
### Usage : 1. Choose a method: Fully, Conv
###         2. Choose rarefaction level: hy, rare
### Author : Zachary
### Date   : 23.01.21
########################################

#import nececccasry libraries
import scipy.io as sio
import numpy as np
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
import torch
import torch.tensor as tensor



# method = "Fully" # one of ["Fully" , "Conv" or "POD"]
# level = "rare" # one of ["hy", "rare"]
# iv = 1

#load the full order BGK data
#############################

def load_BGKandMethod():
    if (method == 'Fully' or method=="POD") and level == 'hy' and train == 'No':
        c = np.load('Data/sod25Kn0p00001_2D_unshuffled.npy')
    elif (method == 'Fully' or method=="POD") and level == 'hy' and train == 'Yes':
        c = np.load('Data/sod25Kn0p00001_2D.npy')
    elif (method == 'Fully' or method=="POD") and level == 'rare' and train == 'No':
        c = np.load('Data/sod25Kn0p01_2D_unshuffled.npy')
    elif (method == 'Fully' or method=="POD") and level == 'rare' and train == 'Yes':
        c = np.load('Data/sod25Kn0p01_2D.npy')
    elif method == 'Conv' and level == 'hy' and train == 'No':
        c = np.load('Data/sod25Kn0p00001_4D_unshuffled.npy')
    elif method == 'Conv' and level == 'hy' and train == 'Yes':
        c = np.load('Data/sod25Kn0p00001_4D.npy')
    elif method == 'Conv' and level == 'rare' and train == 'No':
        c = np.load('Data/sod25Kn0p01_4D_unshuffled.npy')   
    else:
        c = np.load('Data/sod25Kn0p01_4D.npy')

    print("Method:",method,"Level:",level, "Train:",train)
    v = sio.loadmat('Data/sod25Kn0p01/v.mat')
    t = sio.loadmat('Data/sod25Kn0p01/t.mat')
    x = sio.loadmat('Data/sod25Kn0p01/x.mat')
    x = x['x']
    v = v['v']
    t  = t['treport']
    x = x.squeeze()
    t=t.squeeze()
    t=t.T

    return x,v,t,c





#evaluate the models
####################

# train = "No"    # We don't need to train, the models are already trained
# x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
# if method == "Fully":
#   from FullyConnected import model # import the method
#   c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
#   rec, code = model(c)
#   c = c.detach().numpy()
#   rec = rec.detach().numpy()
# elif method == "Conv":
#   from Convolutional import model
#   c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
#   rec, code = model(c)
#   c = c.detach().numpy()
#   rec = rec.detach().numpy()
# else:
#   from POD import pod as model
#   rec, code = model(c)
        
# l2 = np.linalg.norm((c - rec).flatten())/np.linalg.norm(c.flatten()) # calculatre L2-Norm Error
# print('L2-Norm Error =',l2)

#Variation of intrinsic variables
#################################
train = "Yes"   # We want to train the models again and change the int. vars.
int_vars = [1,2,4,8,16,32] # number of intrinsic varibales to check

for level in ["hy", "rare"]:
  for iv in int_vars:
          # method = "Conv"
          # x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
          # from Convolutional import model_train
          # c = tensor(c,dtype=torch.float)
          # model_train(c,iv,level)

            method = "Fully"
            x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
            from FullyConnected import model_train
            c = tensor(c,dtype=torch.float)
            model_train(c,iv,level)

#Plot the Bad Mistakes
######################

# def shapeback_field(predict):
#     f = np.empty([25,40,200])
#     n = 0
#     for i in range(25):
#         for j in range(200):
#             f[i,:,j] = predict[j+n,:]
#         n += 200
#     return(f) # shaping back the field # Shape the reconstruction from 5000x40 bach to 25x40x200

# train = "No"
# x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
# from FullyConnected import model
# c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
# rec, code = model(c)
# c = c.detach().numpy()
# rec = rec.detach().numpy()
# rec = shapeback_field(rec)
# c = shapeback_field(c)
# a = c-rec
# a = np.sum(a,axis=(1,2))


# plt.plot(a)
# plt.show()










