'''
SVD PLOTS for BA
'''

import scipy.io as sio
import numpy as np
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

import tikzplotlib

num_mod = 3
qty = "hy" #["hy" or "rare"]

#Load Data

v = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/t.mat')
x = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/x.mat')
x = x['x']
v = v['v']
t  = t['treport']
x = x.squeeze()
t=t.squeeze()
t=t.T

if qty == "hy":
	c = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
else:
	c = np.load('/home/zachi/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p01_2D_unshuffled.npy')
c=c.T


#SVD

u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues

S = np.diagflat(s)
xx = u[:,:3]@S[:3,:3]@vh[:3,:]

print(u.shape)
def plot_pod_modes(v,u):
	fig, ax = plt.subplots(num_mod,1)
	for i in range(num_mod):
		ax[i].plot(v,u[:,i],'k')
		ax[i].set_xlabel('v')
		ax[i].set_ylabel('gamma{}'.format(i))
	tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/Results/Hydro/PODModes.tex')
	plt.show()


def plot_cumu():


	#Plot Cumultative Energy and Singular Vaules
	k = range(len(s))
	plt.figure(1)
	plt.subplot(1,2,1)
	plt.semilogy(k,s,'.-''k')
	plt.ylabel('sigma')
	plt.xlabel('k')
	plt.subplot(1,2,2)
	plt.plot(k,np.cumsum(s)/np.sum(s),'.-''k')
	plt.ylabel('Cumultative Energy')
	plt.xlabel('k')
	tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/SVD/CumSum_Hydro.tex')
	plt.show()

	return

predict = xx
test_error = norm((c[:] - predict[:]).flatten())/norm(c[:].flatten())
print(test_error)

plot_pod_modes(v,u)