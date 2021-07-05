import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
##import tikzplotlib


from os.path import join
from pathlib import Path
home = Path.home()
loc_data = "rom-using-autoencoders/01_Thesis/python/Chapter_5"

#load the full order BGK data
def load_BGKandMethod(method, level):
    if (method == 'Fully' or method=="POD") and level == 'hy':
        c = np.load(join(home,loc_data,
            'Preprocessed_Data/sod25Kn0p00001_2D_unshuffled.npy'))

    elif (method == 'Fully' or method=="POD") and level == 'rare':
        c = np.load(join(home,loc_data,
            'Preprocessed_Data/sod25Kn0p01_2D_unshuffled.npy'))

    elif method == 'Conv' and level == 'hy':
        c = np.load(join(home,loc_data,
            'Preprocessed_Data/sod25Kn0p00001_4D_unshuffled.npy'))

    elif method == 'Conv' and level == 'rare':
        c = np.load(join(home,loc_data,
            'Preprocessed_Data/sod25Kn0p01_4D_unshuffled.npy'))   

    return c

v = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/v.mat"))
v = v['v']

def macro(f):
    dv = 0.51282051
    rho = np.sum(f,axis = 1) * dv
    rhou = f * v
    rhou = np.sum((rhou),axis = 1) * dv
    E = f * ((v**2) * .5) 
    E = np.sum(E, axis = 1) * dv 
    E = rhou / rho
    return(rho, rhou, E,)

def conservation(macro):
    rho, rhou, E = macro
    dtrho = np.gradient(np.sum(rho,axis=1))
    dtrhou = np.gradient(np.sum(rhou,axis=1))
    dtE = np.gradient(np.sum(E,axis=1))
    return (dtrho, dtrhou, dtE)

def shapeback_field(c):  #Shape the reconstruction from 5000x40 bach to 25x40x200
    t = int(c.shape[0]/200)
    f = np.empty([t,40,200])
    n = 0
    for i in range(t):
        for j in range(200):
            f[i,:,j] = c[j+n,:]
        n += 200
    return(f)

fig,ax = plt.subplots(2,3,tight_layout=True) # for macroscopic quantities
fig.suptitle("rho, rho u and E from FOM, POD, the FCNNs, and the CNN at t=0.12s")
figg,axxs = plt.subplots(2,3,tight_layout=True) # for conservation
figg.suptitle("Consevation for Hy top (rare) (bottom) from FOM, POD,The FCNNs, and the CNN")

for idx, level in enumerate(["hy","rare"]):

    method = "POD"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from POD import pod 
    rec, z = pod(c, level)

    rec_pod = shapeback_field(rec)
    c = shapeback_field(c)

    m_pod = macro(rec_pod)
    c_pod= conservation(m_pod)
    
    method = "Fully"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from FullyConnected import fully
    rec, z = fully(c, level)

    rec = rec.detach().numpy()
    rec_fully = shapeback_field(rec)
    c = shapeback_field(c)

    m_fully = macro(rec_fully)
    c_fully = conservation(m_fully)

    method = "Conv"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from Conv import conv
    rec, z = conv(c)

    rec = rec.detach().numpy()
    rec_conv = np.swapaxes(rec.squeeze(),0,1)
    c = np.swapaxes(c.squeeze(),0,1)

    m_conv = macro(rec_conv)
    c_conv = conservation(m_conv)


    m_fom = macro(c)
    c_fom = conservation(m_fom)
    

    x = np.linspace(start=0,stop=0.995,num=200)
    t = np.linspace(start=0,stop=0.12,num=25)


    ax[idx,0].plot(x,m_fom[0][-1],'k''-x',label='FOM',markevery=5,markersize=5)
    ax[idx,0].plot(x,m_pod[0][-1],'r''-o',label='POD',markevery=5,markersize=5)
    ax[idx,0].plot(x,m_fully[0][-1],'p''--',label='FCNN',markevery=5,markersize=5)
    ax[idx,0].plot(x,m_conv[0][-1],'g''-v',label='CNN',markevery=5,markersize=5)
    ax[idx,0].set_ylabel('rho')
    ax[idx,0].set_xlabel('x')
    ax[idx,0].legend()
    ax[idx,1].plot(x,m_fom[1][-1],'k''-x',label='FOM',markevery=5,markersize=5)
    ax[idx,1].plot(x,m_pod[1][-1],'r''-o',label='POD',markevery=5,markersize=5)
    ax[idx,1].plot(x,m_fully[1][-1],'p''--',label='FCNN',markevery=5,markersize=5)
    ax[idx,1].plot(x,m_conv[1][-1],'g''-v',label='CNN',markevery=5,markersize=5)
    ax[idx,1].set_ylabel('rho u')
    ax[idx,1].set_xlabel('x')
    ax[idx,1].legend()
    ax[idx,2].plot(x,m_fom[2][-1],'k''-x',label='FOM',markevery=5,markersize=5)
    ax[idx,2].plot(x,m_pod[2][-1],'r''-o',label='POD',markevery=5,markersize=5)
    ax[idx,2].plot(x,m_fully[2][-1],'p''--',label='FCNN',markevery=5,markersize=5)
    ax[idx,2].plot(x,m_conv[2][-1],'g''v',label='CNN',markevery=5,markersize=5)
    ax[idx,2].set_ylabel('E')
    ax[idx,2].set_xlabel('x')
    ax[idx,2].legend()

    axxs[idx,0].plot(t,c_fom[0],'k''-x',label='FOM',markevery=5,markersize=5)
    axxs[idx,0].plot(t,c_pod[0],'r''-o',label='POD',markevery=5,markersize=5)
    axxs[idx,0].plot(t,c_fully[0],'p''--',label='FCNN',markevery=5,markersize=5)
    axxs[idx,0].plot(t,c_conv[0],'g''-v',label='CNN',markevery=5,markersize=5)
    axxs[idx,0].set_ylabel('rho')
    axxs[idx,0].set_xlabel('t')
    axxs[idx,0].legend()
    axxs[idx,1].plot(t,c_fom[1],'k''-x',label='FOM',markevery=5,markersize=5)
    axxs[idx,1].plot(t,c_pod[1],'r''-o',label='POD',markevery=5,markersize=5)
    axxs[idx,1].plot(t,c_fully[1],'p''--',label='FCNN',markevery=5,markersize=5)
    axxs[idx,1].plot(t,c_conv[1],'g''-v',label='CNN',markevery=5,markersize=5)
    axxs[idx,1].set_ylabel('rho u')
    axxs[idx,1].set_xlabel('t')
    axxs[idx,1].legend()
    axxs[idx,2].plot(t,c_fom[2],'k''-x',label='FOM',markevery=5,markersize=5)
    axxs[idx,2].plot(t,c_pod[2],'r''-o',label='POD',markevery=5,markersize=5)
    axxs[idx,2].plot(t,c_fully[2],'p''--',label='FCNN',markevery=5,markersize=5)
    axxs[idx,2].plot(t,c_conv[2],'g''-v',label='CNN',markevery=5,markersize=5)
    axxs[idx,2].set_ylabel('E')
    axxs[idx,2].set_xlabel('t')
    axxs[idx,2].legend()

#tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_5/MacroError_test.tex'))###

##tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_5/Conservation_test.tex'))###
plt.show()