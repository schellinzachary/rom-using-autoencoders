import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=17) 
plt.rc('ytick', labelsize=17) 

'''
Macroscopic quantities
'''

#load Data

kn0p01 = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p01/f.mat')
kn0p00001 = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/f.mat')
v = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/v.mat')
kn0p01 = kn0p01['f']
kn0p00001 = kn0p00001['f']
v = v['v']

#def macroscopic calculations
def macroscopic(f,v):
	dv = v[1] - v[0]
	rho = np.sum(f,axis=1) * dv
	rho_u = np.sum(f * v,axis=1) * dv
	E = np.sum(f * .5 * v**2,axis=1) * dv
	return rho, rho_u, E

rho_h, rho_u_h, E_h = macroscopic(kn0p00001,v)

rho_g, rho_u_g, E_g = macroscopic(kn0p01,v)

#plot the results
t = -1
fix,ax = plt.subplots(1,3)

ax[0].plot(rho_h[t],'--''k',label='Kn=0.01')
ax[0].plot(rho_g[t],'-''k',label='Kn=0.00001')
ax[0].legend()
ax[0].set_xlabel(r'x',fontsize=17)
ax[0].set_ylabel(r'$\rho$',fontsize=17)
ax[1].plot(rho_u_h[t],'--''k',label='Kn=0.01')
ax[1].plot(rho_u_g[t],'-''k',label='Kn=0.00001')
ax[1].legend()
ax[1].set_xlabel(r'x',fontsize=17)
ax[1].set_ylabel(r'$\rho u$',fontsize=17)
ax[2].plot(E_h[t],'--''k',label='Kn=0.01')
ax[2].plot(E_g[t],'-''k',label='Kn=0.00001')
ax[2].legend()
ax[2].set_xlabel(r'x',fontsize=17)
ax[2].set_ylabel(r'E',fontsize=17)
#plt.savefig('Macroscopic_quantities',tightlayout=True)
plt.show()

