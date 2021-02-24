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
import tikzplotlib
from pathlib import Path
home = str(Path.home())

qty = "hy" #["hy" or "rare"]


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
	u, s, vh = np.linalg.svd(c.T,full_matrices=True) #s Singularvalues
	S = np.diagflat(s)
	for i in range(len(s)):
		rec = u[:,:i]@S[:i,:i]@vh[:i,:]
		l2 = np.linalg.norm((c.T - rec).flatten())/np.linalg.norm(c.T.flatten()) # calculatre L2-Norm Error
		ls.append(l2)
	return s, ls

#Plot cumultative energy and singular values
###########################################
c_r = load_data("hy")
c_h = load_data("rare")
s_r, ls_r = POD(c_r)
s_h, ls_h = POD(c_h)
k = range(len(s_h))
fig, ax = plt.subplots(2,1)
lvl = ["hy","rare"]
for idx, frac in enumerate([[s_h,ls_h],[s_r,ls_r]]):
	ax[0].semilogy(k,frac[0],'.-''k')
	ax[0].set_ylabel('sigma')
	ax[0].set_xlabel('k')
	ax[1].plot(k,np.cumsum(frac[0])/np.sum(frac[0]),'.-''r')
	ax[1].set_ylabel('Cumultative Energy')
	ax[1].set_xlabel('k')
	#tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/SVD/CumSum_Hydro.tex')
	for i in k:
		print(lvl[idx],i, frac[1][i])
plt.show()