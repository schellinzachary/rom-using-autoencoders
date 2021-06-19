import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib

#plt.style.use("seaborn")

from os.path import join
from pathlib import Path
home = Path.home()

v = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/v.mat"))
v = v['v']

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

def macro(f):
    dv = 0.51282051
    rho = np.sum(f,axis = 1) * dv
    rhou = f * v
    rhou = np.sum((rhou),axis = 1) * dv
    E = f * ((v**2) * .5) 
    E = np.sum(E, axis = 1) * dv
    u = rhou / rho
    T = ((2*E) / (3*rho)) - ((u**2)/3) 
    p = rho * T
    return(rho, rhou, E, T, u, p)

def shapeback_field(c):  #Shape the reconstruction from 5000x40 bach to 25x40x200
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

fig_h, hyax = plt.subplots(2,3)
#fig_r, rarax = plt.subplots(2,5)
#fig_f, fomax = plt.subplots(1,5)

for idx, level in enumerate(["hy","rare"]):

    # method = "POD"
    # c = load_BGKandMethod(method, level) # load FOM data for evaluation
    # from POD import pod
    # rec_pod, z_pod = pod(c,level)
    # print(z_pod.shape)
    # z_pod = shapeback_code(z_pod)


    method = "Fully"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from FullyConnected import fully
    rec_fully, z = fully(c, level)
    z = z.detach().numpy()
    z_fcnn = shapeback_code(z)
    c_fom = shapeback_field(c)



    method = "Conv"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from Conv import conv
    rec_conv, z = conv(c)
    z_cnn = z.detach().numpy().squeeze()

    names = ["rho","rhou","E","T","p"]
    
    m_fom = macro(c_fom)
    

    if level == "hy":
        for i in range(3):
            hy = hyax[0,i].imshow(z_fcnn[:,i,:],
                cmap='gray',label="FCNN",
                extent=[0,1,0,0.12],
                aspect="auto",
                origin="lower"
                )
            hyax[0,i].set_xlabel("\(x\)")
            hyax[0,i].set_ylabel("\(t\)")
            h_bar = fig_h.colorbar(hy, ax=hyax[0,i],
                #orientation="horizontal",
                location="top"
                )
            h_bar.set_label("\(z_%s\)"%i)

            hyax[1,i].plot(v,z_cnn[:,i])
            hyax[1,i].set_xlabel("\(v\)")
            hyax[1,i].set_ylabel("\(z_%s\)"%i)
tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/code_hy.tex"))
plt.show()
    # if level == "rare":
    #     exit()
    #     for i in range(5):
    #         fc = rarax[0,i].imshow(z_fcnn[:,i,:],
    #             cmap='gray',label="FCNN",
    #             extent=[0,1,0,0.12],
    #             aspect="auto",
    #             origin="lower"
    #             )
    #         rarax[0,i].set_xlabel("\(x\)")
    #         rarax[0,i].set_ylabel("\(t\)")
    #         rar_bar = fig_r.colorbar(fc, ax=rarax[0,i],
    #             #orientation="horizontal",
    #             location="top"
    #             )
    #         rar_bar.set_label("\(z_%s\)"%i)
    #         rarax[1,i].plot(z_cnn[:,i])
    #         rarax[1,i].set_xlabel("\(v\)")
    #         rarax[1,i].set_ylabel("\(z_%s\)"%i)

    #         fom = fomax[i].imshow(m_fom[i],
    #             cmap='gray',label="FCNN",
    #             extent=[0,1,0,0.12],
    #             aspect="auto",
    #             origin="lower"
    #             )
    #         fomax[i].set_xlabel("\(x\)")
    #         fomax[i].set_ylabel("\(t\)")
    #         fom_bar = fig_f.colorbar(fom, ax=fomax[i],
    #             orientation="horizontal",
    #             location="top"
    #             )
    #         fom_bar.set_label("\(z_%s\)"%i)








#tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/code_rare.tex"))
#tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/macro_all.tex'))
# plt.tight_layout()
# plt.show()


