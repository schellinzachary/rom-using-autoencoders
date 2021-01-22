'''
V-X plot over time
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
#plt.rc('xtick', labelsize=15)
#plt.rc('ytick', labelsize=15)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

fig , (ax1,ax2,ax3) = plt.subplots(nrows = 3)

#Load Data
fa = sio.loadmat('sod25Kn0p01/f.mat')
fb = sio.loadmat('sod25Kn0p00001/f.mat')
fc = sio.loadmat('sod241Kn0p00001/f.mat')

fa = fa['f']
fb = fb['f']
fc = fc['f']

fa = fa[-1,:,:] - fa[0,:,:]
fb = fb[-1,:,:] - fb[0,:,:]
fc = fc[-1,:,:] - fc[0,:,:]
X = np.linspace(-10,10,40)
Y = np.linspace(0,1,200)
Y,X = np.meshgrid(Y,X)


p1 = ax1.pcolor(Y,X,fa)
fig.colorbar(p1, ax=ax1)
ax1.set_title('Kn 0.01, 25 snapshots')
plt.ylabel(r'v')
plt.xlabel(r'x')

p2 = ax2.pcolor(Y,X,fb)
fig.colorbar(p2,ax=ax2)
ax2.set_title('Kn 0.00001, 25 snapshots')
plt.ylabel(r'v')
plt.xlabel(r'x')

p3 = ax3.pcolor(Y,X,fc)
fig.colorbar(p3,ax=ax3)
ax3.set_title('Kn 0.00001, 241 snapshots')

plt.ylabel(r'v', fontsize = '15')
plt.xlabel(r'x', fontsize = '15')




fig.tight_layout()
plt.show()