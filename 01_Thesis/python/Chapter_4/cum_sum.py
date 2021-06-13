### Producing the plots for for Chapter 4
###
### Usage : 1. Choose rarefaction level: hy, rare
###         
### Author : Zachary
### Date   : 24.02.21
########################################

import scipy.io as sio
import numpy as np
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
#import tikzplotlib
from pathlib import Path
home = str(Path.home())

#qty = "hy" #["hy" or "rare"]


#Load Data
##########
def load_data(qty):
	if qty == "hy":
		c = np.load('%s/rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy'%home)
	else:
		c = np.load('%s/rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy'%home)
	return(c)

#Perform POD
############
def POD(c):
	ls = []
	u, s, vh = np.linalg.svd(c.T,full_matrices=False) #s Singularvalues
	S = np.diagflat(s)
	for i in range(len(s)):
		rec = u[:,:i]@S[:i,:i]@vh[:i,:]
	return s


#Plot cumultative energy and singular values
############################################
c_h = load_data("hy")
c_r = load_data("rare")
s_r = POD(c_r)
s_h = POD(c_h)

k = range(1,len(s_h)+1)
fig, ax = plt.subplots(1,2)
figs, axs = plt.subplots(1,2)
lvl = ["hy","rare"]
labels = ["o-","-v"]

ax[0].semilogy(k,s_h,labels[0],label='%s'%lvl[0])
ax[0].set_ylabel('sigma')
ax[0].set_xlabel('k')
ax[1].plot(k,np.cumsum(s_h)/np.sum(s_h),labels[0],label='%s'%lvl[0])
ax[1].set_ylabel('Cumultative Energy')
ax[1].set_xlabel('k')
ax[0].legend()
ax[1].legend()
### tikzplotlib.save('%s/rom-using-autoencoders/01_Thesis/Figures/SVD/CumSum_test_hy.tex'%home)

axs[0].semilogy(k,s_r,labels[1],label='%s'%lvl[1])
axs[0].set_ylabel('sigma')
axs[0].set_xlabel('k')
axs[1].plot(k,np.cumsum(s_r)/np.sum(s_r),labels[1],label='%s'%lvl[1])
axs[1].set_ylabel('Cumultative Energy')
axs[1].set_xlabel('k')
axs[0].legend()
axs[1].legend()
###tikzplotlib.save('%s/rom-using-autoencoders/01_Thesis/Figures/SVD/CumSum_test_rare.tex'%home)
plt.show()
