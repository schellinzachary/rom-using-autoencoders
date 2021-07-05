import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sn
###import tikzplotlib

#plt.style.use("seaborn")

from os.path import join
from pathlib import Path
home = Path.home()
loc_data = "rom-using-autoencoders/01_Thesis/python/Chapter_5"

v = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/v.mat"))
x = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/x.mat"))

v = v['v']
x = x['x'].squeeze()

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

fig_h, hyax = plt.subplots(3,3,tight_layout=True)
fig_r, rarax = plt.subplots(3,5,tight_layout=True)
fig_fr, fomarx = plt.subplots(3,5,tight_layout=True)
fig_fh, fomahx = plt.subplots(3,5,tight_layout=True)
fig_cnp, cnpax = plt.subplots(4,5,tight_layout=True)


fig_fh.suptitle("FOM Macro. qty hydro")
fig_fr.suptitle("FOM Macro. qty rare")
fig_h.suptitle("Code FCNN  hydro")
fig_r.suptitle("Code FCNN  rare")
fig_cnp.suptitle("Code CNN and POD hy(top)&rare(bottom)")


for idx, level in enumerate(["hy","rare"]):

    method = "POD"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from POD import pod
    rec_pod, z_pod = pod(c.T,level)
    #z_pod = shapeback_code(z_pod)



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

    names = ["rho","rhou","E","T","u"]
    m_fom = macro(c_fom)
    

    # if level == "rare":

    #     df_mfom = np.stack(m_fom,axis=1)
    #     df_mfom = pd.DataFrame(df_mfom[-1,:,:].T,columns=names)
    #     df_zfcnn = pd.DataFrame(z_fcnn[-1,:,:].T,columns=["1","2","3","4","5"])
    #     df = pd.concat([df_mfom,df_zfcnn],axis=1)
    #     corr = df.corr()
        
    #     sn.heatmap(corr.iloc[:5,5:], annot=True)
    #     plt.show()

    # exit()

    if level == "hy":
        for i in range(5):
            fom = fomahx[0,i].imshow(m_fom[i],
                cmap='gray',label="FCNN",
                extent=[0,1,0,0.12],
                aspect="auto",
                origin="lower",
                )
            fomahx[0,i].set_xlabel("\(x\)")
            fomahx[0,i].set_ylabel("\(t\)")
            fomahx[0,i].set_yticks([0,0.12])
            fomahx[0,i].set_xticks([0,1])
            fom_bar = fig_fh.colorbar(fom, ax=fomahx[0,i],
                orientation="horizontal",
                location="top",
                ticks=[np.min(m_fom[i]), np.max(m_fom[i])]
                )
            fom_bar.set_label("\(%s\)"%names[i])

            fomahx[1,i].plot(x,m_fom[i][11,:])
            fomahx[1,i].set_xlabel("\(x\)")
            fomahx[1,i].set_ylabel("\(%s\)"%names[i])
            fomahx[1,i].set_yticks([np.min(m_fom[i][11,:]),np.max(m_fom[i][11,:])])
            fomahx[1,i].set_xticks([0,1])

            fomahx[2,i].plot(x,m_fom[i][-1,:])
            fomahx[2,i].set_xlabel("\(x\)")
            fomahx[2,i].set_ylabel("\(%s\)"%names[i])
            fomahx[2,i].set_yticks([np.min(m_fom[i][-1,:]),np.max(m_fom[i][-1,:])])
            fomahx[2,i].set_xticks([0,1])
    #     # tikzplotlib.save(join(home,
    #     #     "rom-using-autoencoders/01_Thesis/Figures/Chapter_5/fom_mac_hy.tex"),strict=True)
    # #     # exit()
    if level == "rare":
        for i in range(5):
            fom = fomarx[0,i].imshow(m_fom[i],
                cmap='gray',label="FCNN",
                extent=[0,1,0,0.12],
                aspect="auto",
                origin="lower"
                )
            fomarx[0,i].set_xlabel("\(x\)")
            fomarx[0,i].set_ylabel("\(t\)")
            fomarx[0,i].set_yticks([0,0.12])
            fomarx[0,i].set_xticks([0,1])
            fom_bar = fig_fr.colorbar(fom, ax=fomarx[0,i],
                #orientation="horizontal",
                location="top",
                ticks=[np.min(m_fom[i]), np.max(m_fom[i])]
                )
            fom_bar.set_label("\(%s\)"%names[i])

            fomarx[1,i].plot(x,m_fom[i][11,:])
            fomarx[1,i].set_xlabel("\(x\)")
            fomarx[1,i].set_ylabel("\(%s\)"%names[i])
            fomarx[1,i].set_yticks([np.min(m_fom[i][11,:]),np.max(m_fom[i][11,:])])
            fomarx[1,i].set_xticks([0,1])

            fomarx[2,i].plot(x,m_fom[i][-1,:])
            fomarx[2,i].set_xlabel("\(x\)")
            fomarx[2,i].set_ylabel("\(%s\)"%names[i])
            fomarx[2,i].set_yticks([np.min(m_fom[i][-1,:]),np.max(m_fom[i][-1,:])])
            fomarx[2,i].set_xticks([0,1])
    #     # tikzplotlib.save(join(home,
    #     #     "rom-using-autoencoders/01_Thesis/Figures/Chapter_5/fom_mac_rare.tex"),strict=True)
    #     plt.show()
    #     exit()

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
            hyax[0,i].set_yticks([0,0.12])
            hyax[0,i].set_xticks([0,1])
            h_bar = fig_h.colorbar(hy, ax=hyax[0,i],
                #orientation="horizontal",
                location="top",
                ticks=[np.min(z_fcnn[:,i,:]), np.max(z_fcnn[:,i,:])]
                )
            h_bar.set_label("\(z_%s\)"%i)

            hyax[1,i].plot(x,z_fcnn[11,i,:])
            hyax[1,i].set_xlabel("\(x\)")
            hyax[1,i].set_ylabel("\(z_%s\)"%i)
            hyax[1,i].set_yticks([np.min(z_fcnn[11,i,:]),np.max(z_fcnn[11,i,:])])
            hyax[1,i].set_xticks([0,1])

            hyax[2,i].plot(x,z_fcnn[-1,i,:])
            hyax[2,i].set_xlabel("\(x\)")
            hyax[2,i].set_ylabel("\(z_%s\)"%i)
            hyax[2,i].set_yticks([np.min(z_fcnn[-1,i,:]),np.max(z_fcnn[-1,i,:])])
            hyax[2,i].set_xticks([0,1])

            cnpax[2,i].plot(v,z_pod[:,i])
            cnpax[2,i].set_xlabel("\(v\)")
            cnpax[2,i].set_ylabel("\(z_%s\)"%i)
            cnpax[2,i].set_yticks([np.min(z_pod[:,i]),np.max(z_pod[:,i])])
            cnpax[2,i].set_xticks([-10,10])
        # tikzplotlib.save(join(home,
        #     "rom-using-autoencoders/01_Thesis/Figures/Chapter_5/code_hydrot.tex"),strict=True)
        # plt.show()
        # exit()
    if level == "hy":
        for i in range(5):
            cnpax[0,i].plot(v,z_cnn[:,i])
            cnpax[0,i].set_xlabel("\(v\)")
            cnpax[0,i].set_ylabel("\(z_%s\)"%i)
            cnpax[0,i].set_yticks([np.min(z_cnn[:,i]),np.max(z_cnn[:,i])])
            cnpax[0,i].set_xticks([-10,10])
    if level == "rare":
        for i in range(5):
            fc = rarax[0,i].imshow(z_fcnn[:,i,:],
                cmap='gray',label="FCNN",
                extent=[0,1,0,0.12],
                aspect="auto",
                origin="lower"
                )
            rarax[0,i].set_xlabel("\(x\)")
            rarax[0,i].set_ylabel("\(t\)")
            rarax[0,i].set_yticks([0,0.12])
            rarax[0,i].set_xticks([0,1])
            rar_bar = fig_r.colorbar(fc, ax=rarax[0,i],
                #orientation="horizontal",
                location="top",
                ticks=[np.min(z_fcnn[:,i,:]), np.max(z_fcnn[:,i,:])]
                )
            rar_bar.set_label("\(z_%s\)"%i)

            rarax[1,i].plot(x,z_fcnn[11,i,:])
            rarax[1,i].set_xlabel("\(x\)")
            rarax[1,i].set_ylabel("\(z_%s\)"%i)
            rarax[1,i].set_yticks([np.min(z_fcnn[11,i,:]),np.max(z_fcnn[11,i,:])])
            rarax[1,i].set_xticks([0,1])

            rarax[2,i].plot(x,z_fcnn[-1,i,:])
            rarax[2,i].set_xlabel("\(x\)")
            rarax[2,i].set_ylabel("\(z_%s\)"%i)
            rarax[2,i].set_yticks([np.min(z_fcnn[-1,i,:]),np.max(z_fcnn[-1,i,:])])
            rarax[2,i].set_xticks([0,1])

            
            cnpax[1,i].plot(v,z_cnn[:,i])
            cnpax[1,i].set_xlabel("\(v\)")
            cnpax[1,i].set_ylabel("\(z_%s\)"%i)
            cnpax[1,i].set_yticks([np.min(z_cnn[:,i]),np.max(z_cnn[:,i])])
            cnpax[1,i].set_xticks([-10,10])

            cnpax[3,i].plot(v,z_pod[:,i])
            cnpax[3,i].set_xlabel("\(v\)")
            cnpax[3,i].set_ylabel("\(z_%s\)"%i)
            cnpax[3,i].set_yticks([np.min(z_pod[:,i]),np.max(z_pod[:,i])])
            cnpax[3,i].set_xticks([-10,10])
        # tikzplotlib.save(join(home,
        #     "rom-using-autoencoders/01_Thesis/Figures/Chapter_5/code_cnn_pod.tex"),strict=True)
        # plt.show()
        # exit()



plt.show()

#tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/code_rare.tex"))
#tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/macro_all.tex'))
# plt.tight_layout()
# plt.show()


