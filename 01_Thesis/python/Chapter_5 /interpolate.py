import numpy as np
import scipy.io as sio
from scipy import interpolate
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib

#plt.style.use("seaborn")

from os.path import join
from pathlib import Path
home = Path.home()

v = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/v.mat"))
x = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/x.mat"))
t = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/t.mat"))
v = v['v']
x = x['x'].squeeze()
t = t['treport'].squeeze()

#load the full order BGK data
def load_BGKandMethod(method, level):
    if (method == 'Fully' or method=="POD") and level == 'hy':
        c = np.load('Preprocessed_Data/sod25Kn0p00001_2D_unshuffled.npy')

    elif (method == 'Fully' or method=="POD") and level == 'rare':
        c = np.load('Preprocessed_Data/sod25Kn0p01_2D_unshuffled.npy')

    elif method == 'Conv' and level == 'hy':
        c = np.load('Preprocessed_Data/sod25Kn0p00001_4D_unshuffled.npy')

    elif method == 'Conv' and level == 'rare':
        c = np.load('Preprocessed_Data/sod25Kn0p01_4D_unshuffled.npy')   

    return c
def load_BGK_241():
    c = np.load('Data/sod241Kn0p00001_2D_unshuffled.npy')
    return(c)

def macro(f):
    dv = 0.51282051
    rho = np.sum(f,axis = 1) * dv
    rhou = f * v
    rhou = np.sum((rhou),axis = 1) * dv
    E = f * ((v**2) * .5) 
    E = np.sum(E, axis = 1) * dv
    u = rhou / rho
    T = ((2*E) / (3*rho)) - ((u**2)/3) 
    return(rho, rhou, E, T, u)

def shapeback_field(c):
    t = int(c.shape[0]/200)
    f = np.empty([t,40,200])
    n = 0
    for i in range(t):
        for j in range(200):
            f[i,:,j] = c[j+n,:]
        n += 200
    return(f)

def shapeback_code(z):
    c = np.empty((25,z.shape[1],200))
    n=0
    for i in range(25):
        for p in range(200):
          c[i,:,p] = z[p+n,:]
        n += 200
    return(c) # shaping back the code

def shape_AE_code(g):
    c = np.empty((g.shape[0]*g.shape[2],g.shape[1]))
    for i in range(g.shape[1]):
        n = 0
        for t in range(g.shape[0]):
          c[n:n+200,i] = g[t,i,:]
          n += 200
    return(c)

for idx, level in enumerate(["hy"]):

	method="Fully"
	c = load_BGKandMethod(method, level) # load FOM data for evaluation
	from FullyConnected import fully

	rec, code = fully(c, level)
	code = code.detach().numpy()
	code = shapeback_code(code)

	ti=241
	iv=3
	cnew=np.empty([ti,iv,200])
	tnew=np.linspace(0.0,0.12,num=ti)

	for i in range(iv):
	    f = interpolate.interp1d(t[::1],code[::1,i,:],axis=0,kind='cubic')
	    cnew[:,i,:]=f(tnew)
	#print(np.sum(np.abs(cnew)-np.abs(code)))



	codenew = shape_AE_code(cnew)
	#codenew=tensor(codenew,dtype=torch.float)

	from FullyConnected import decoder
	fnew = decoder(codenew, level)
	fnew=fnew.detach().numpy()
	fnew = shapeback_field(fnew)
	#fold = c.detach().numpy()
	fold = np.load('Preprocessed_Data/sod241Kn0p00001_2D_unshuffled.npy')
	fold = shapeback_field(fold)
	l2 = np.linalg.norm((fnew - fold).flatten())/np.linalg.norm(fold.flatten()) # calculatre L2-Norm Error
	print('Interpolation L2 Error:',l2)

m_old = macro(fold)
m_new = macro(fnew)
fig, axs = plt.subplots(1,3)

axs[0].plot(x,m_new[0][-1],'-o',label='prediction')
axs[0].plot(x,m_old[0][-1],'k+',label='ground truth')
axs[0].set_xlabel('\(x\)')
axs[0].set_ylabel('\(\rho\)')
axs[0].legend()
axs[1].plot(x,m_new[1][-1],'-o',label='prediction')
axs[1].plot(x,m_old[1][-1],'k+',label='ground truth')
axs[1].set_xlabel('\(x\)')
axs[1].set_ylabel('\(\rho u\)')
axs[1].legend()
axs[2].plot(x,m_new[2][-1],'-o',label='prediction')
axs[2].plot(x,m_old[2][-1],'k+',label='ground truth')
axs[2].set_xlabel('\(x\)')
axs[2].set_ylabel('\(E\)')
axs[2].legend()
tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/Hy_Intt.tex"))
plt.show()
