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
import torch.tensor as tensor
import torch



method = "Conv" # one of ["Fully" , "Conv"]
level = "rare" # one of ["hy", "rare"]


#load the full order BGK data
#############################

def load_BGKandMethod():
	if method == "Fully" and level == "hy":
		c = np.load('Data/sod25Kn0p00001_2D_unshuffled.npy')
	elif method == "Fully" and level == "rare":
		c = np.load('Data/sod25Kn0p01_2D_unshuffled.npy')
	elif method == "Conv" and level == "hy":
		c = np.load('Data/sod25Kn0p00001_4D_unshuffled.npy')
	else:
		c = np.load('Data/sod25Kn0p01_4D_unshuffled.npy')		
	print("Method:",method,"Level:",level)
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
x,v,t,c = load_BGKandMethod()

if method == "Fully":
	from FullyConnected import model # import the method
else:
	from Convolutional import model

c = tensor(c,dtype=torch.float) # make c a tensor

rec, code = model(c)

#L2 Error
#########
l2 = torch.norm((c - rec).flatten())/torch.norm(c.flatten())
print('L2',l2)





