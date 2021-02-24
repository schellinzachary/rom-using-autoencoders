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
	u, s, vh = np.linalg.svd(c.T,full_matrices=True) #s Singularvalues
	return(s)

#Plot cumultative energy and singular values
###########################################
c_r = load_data("hy")
c_h = load_data("rare")
s_r = POD(c_r)
s_h = POD(c_h)
print(s_h)
k = range(len(s_h))
fig, ax = plt.subplots(2,2)
for idx, frac in enumerate([s_h,s_r]):
	ax[0,0].semilogy(k,frac,'.-''k')
	ax[0,0].ylabel('sigma')
	ax[0,0].xlabel('k')
	plt.plot(k,np.cumsum(s)/np.sum(s),'.-''r')
	plt.ylabel('Cumultative Energy')
	plt.xlabel('k')
	#tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/SVD/CumSum_Hydro.tex')
	plt.show()