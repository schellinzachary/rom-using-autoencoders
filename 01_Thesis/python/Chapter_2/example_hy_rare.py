import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
##import tikzplotlib


from os.path import join
from pathlib import Path
home = Path.home()
loc_data = "rom-using-autoencoders/01_Thesis/python/Chapter_2"

hy = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p00001/f.mat"))
hy = hy["f"]
rare = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/f.mat"))
rare = rare["f"]
v = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/v.mat"))
v = v["v"]
x = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/x.mat"))
x = x["x"].squeeze()


def macro(f):
    dv = 0.51282051
    rho = np.sum(f,axis = 1) * dv
    rhou = f * v
    rhou = np.sum((rhou),axis = 1) * dv
    E = f * ((v**2) * .5) 
    E = np.sum(E, axis = 1) * dv 
    #v = rhou / rho
    return(rho, rhou, E,)




fig, ax = plt.subplots(1,3,tight_layout=True)
fig.suptitle("Examples of rho, rho u and E for Hy and Rare at t=0.12s")
figg, axx = plt.subplots(2,3)#,tight_layout=True)
figg.suptitle("Examples of f at t = [0s,0.06s,0.12s]; hy top, rare bottom")
names = ["hy","rare"]
times = [0,13,-1]
for idx, level in enumerate([hy,rare]):

    mac = m_fom = macro(level)

    ax[0].plot(x,mac[0][-1],label="%s"%names[idx])
    ax[0].set_xlabel("\(x\)")
    ax[0].set_ylabel("\(\rho\)")
    ax[0].legend()
    ax[1].plot(x,mac[1][-1],label="%s"%names[idx])
    ax[1].set_xlabel("\(x\)")
    ax[1].set_ylabel("\(\rho u\)")
    ax[1].legend()
    ax[2].plot(x,mac[2][-1],label="%s"%names[idx])
    ax[2].set_xlabel("\(x\)")
    ax[2].set_ylabel("\(E\)")
    ax[2].legend()

    im = axx[idx,0].imshow(level[0,:,:],
                cmap='gray',label="0s",
                extent=[0,1,-10,10],
                aspect="auto",
                origin="lower"
                )
    axx[idx,0].set_xlabel("\(x\)")
    axx[idx,0].set_ylabel("\(v\)")
    axx[idx,1].imshow(level[13,:,:],
                cmap='gray',label="0.06s",
                extent=[0,1,-10,10],
                aspect="auto",
                origin="lower"
                )
    axx[idx,1].set_xlabel("\(x\)")
    axx[idx,1].set_ylabel("\(v\)")

    axx[idx,2].imshow(level[-1,:,:],
                        cmap='gray',label="0.12s",
                extent=[0,1,-10,10],
                aspect="auto",
                origin="lower"
                )
    axx[idx,2].set_xlabel("\(x\)")
    axx[idx,2].set_ylabel("\(v\)")
    figg.colorbar(im, ax=axx[idx])
plt.show()
