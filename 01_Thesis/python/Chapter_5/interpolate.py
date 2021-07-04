import numpy as np
import scipy.io as sio
import pandas as pd
from scipy import interpolate
from scipy.interpolate import BarycentricInterpolator
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
###import tikzplotlib

#plt.style.use("seaborn")

from os.path import join
from pathlib import Path
home = Path.home()
loc_data = "rom-using-autoencoders/01_Thesis/python/Chapter_5"

v = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/v.mat"))
x = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/x.mat"))
t = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod25Kn0p01/t.mat"))
tnew = sio.loadmat(join(home, "rom-using-autoencoders/02_data_sod/sod241Kn0p00001/t.mat"))
loc_datahy = "rom-using-autoencoders/02_data_sod/sod25Kn0p00001/f.mat"
loc_datarare = "rom-using-autoencoders/02_data_sod/sod25Kn0p01/f.mat"
v = v['v']
x = x['x'].squeeze()
t = t['treport'].squeeze()
tnew = tnew['treport'].squeeze()

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

def twoD(f):
  x=f.shape[2]
  t=f.shape[0]
  v=f.shape[1]
  #Submatrix
  c = np.empty((t*x,v))
  n = 0

  #Build 2D-Version
  for i in range(t):                                         
    for j in range(x):
      c[j+n,:]=f[i,:,j]
    n +=200
  return(c)

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

    c = np.empty((int(z.shape[0]/200),z.shape[1],200))
    n=0
    for i in range(int(z.shape[0]/200)):
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

fig, axs = plt.subplots(1,3,tight_layout=True)
#figg, axxs = plt.subplots(1,3,tight_layout=True)
figgg, axxxs = plt.subplots(2,4,tight_layout=True)





del_vars = [2,3,4,5]

from FullyConnected import fully, decoder
from Fully_interpolate import fully_int, decoder_int

ti=241
for idx, level in enumerate(["hy"]):

    if level == "hy":
        iv = 3
        method = "Fully"
        c = load_BGKandMethod(method, level)



        #from FullyConnected import fully, decoder
        rec, code = fully(c, level)
        code = code.detach().numpy()
        code = shapeback_code(code)

        cnew=np.empty([ti,iv,200])
        for i in range(iv):
            f = interpolate.interp1d(t[::1],code[::1,i,:],axis=0,kind='cubic')
            cnew[:,i,:]=f(tnew)


        codenew = shape_AE_code(cnew)

        fnew = decoder(codenew, level)
        fnew=fnew.detach().numpy()
        fnew = shapeback_field(fnew)
        #fold = c.detach().numpy()
        fold = np.load(join(home,loc_data,
            'Preprocessed_Data/sod241Kn0p00001_2D_unshuffled.npy'))
        fold = shapeback_field(fold)
        l2 = np.linalg.norm((fnew - fold).flatten())/np.linalg.norm(fold.flatten()) # calculatre L2-Norm Error

        m_old = macro(fold)
        m_new = macro(fnew)

ti=25
l2_hy = []
l2_rare = []
l2_hy_int = []
l2_rare_int = []
e_hyold = []
e_hynew = []
e_rareold = []
e_rarenew = []
fill = None
kind = "cubic"
for del_var in del_vars:
    if del_var == 5:
        fill = "extrapolate"
        # t = t[:-4]
        # ti = 21
    for idx, level in enumerate(["hy","rare"]):

        if level == "hy":
            iv = 3
            method = "Fully"
            f = sio.loadmat(join(home,loc_datahy)) 
            c  = f['f']
            c_less = twoD(c[::del_var,:,:])

            rec, code = fully_int(del_var, c_less, level)
            l2h = np.linalg.norm((rec.detach().numpy() - c_less).flatten())/np.linalg.norm(c_less.flatten())
            l2_hy.append(l2h)
            code = code.detach().numpy()
            code = shapeback_code(code)
            cnew=np.empty([ti,iv,200])
            for i in range(iv):
                f = interpolate.interp1d(t[::del_var],code[::1,i,:],
                    axis=0,kind=kind,
                    fill_value=fill
                    )
                cnew[:,i,:]=f(t)


            codenew = shape_AE_code(cnew)

            fnew = decoder_int(del_var, codenew, level)
            fnew=fnew.detach().numpy()
            fnew = shapeback_field(fnew)


            fold = c
            # if del_var == 5:
            #     fold = fold[:-4,:,:]
            l2hi = np.linalg.norm((fnew - fold).flatten())/np.linalg.norm(fold.flatten()) # calculatre L2-Norm Error
            l2_hy_int.append(l2hi)

            mi_old = macro(fold)
            e_hyold.append(mi_old[2][-1])
            mi_new = macro(fnew)
            e_hynew.append(mi_new[2][-1])



        if level == "rare":
            iv = 5
            method = "Fully"
            f = sio.loadmat(join(home,loc_datarare)) 
            c  = f['f']
            c_less = twoD(c[::del_var,:,:])

            rec, code = fully_int(del_var, c_less, level)
            l2r = np.linalg.norm((rec.detach().numpy() - c_less).flatten())/np.linalg.norm(c_less.flatten())
            l2_rare.append(l2r)
            code = code.detach().numpy()
            code = shapeback_code(code)

            cnew=np.empty([ti,iv,200])
            for i in range(iv):
                f = interpolate.interp1d(t[::del_var],code[::1,i,:],
                    axis=0,kind=kind,
                    fill_value=fill
                    )
                cnew[:,i,:]=f(t)


            codenew = shape_AE_code(cnew)

            fnew = decoder_int(del_var, codenew, level)
            fnew=fnew.detach().numpy()
            fnew = shapeback_field(fnew)
            #fold = c.detach().numpy()
            fold = c
            # if del_var == 5:
            #     fold = fold[:-4,:,:]
            l2ri = np.linalg.norm((fnew - fold).flatten())/np.linalg.norm(fold.flatten()) # calculatre L2-Norm Error
            l2_rare_int.append(l2ri)

            mir_old = macro(fold)
            e_rareold.append(mir_old[2][-1])
            mir_new = macro(fnew)
            e_rarenew.append(mir_new[2][-1])
l2_list = np.stack([l2_hy,l2_rare,l2_hy_int,l2_rare_int],axis=1)
l2_list = pd.DataFrame(l2_list,
    columns=["l2 hy", "l2 rare","l2 hy int","l2 rare int"])
print(l2_list)




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

# axxs[0].plot(x,mr_new[0][-1],'-o',label='prediction')
# axxs[0].plot(x,mr_old[0][-1],'k+',label='ground truth')
# axxs[0].set_xlabel('\(x\)')
# axxs[0].set_ylabel('\(\rho\)')
# axxs[0].legend()
# axxs[1].plot(x,mr_new[1][-1],'-o',label='prediction')
# axxs[1].plot(x,mr_old[1][-1],'k+',label='ground truth')
# axxs[1].set_xlabel('\(x\)')
# axxs[1].set_ylabel('\(\rho u\)')
# axxs[1].legend()
# axxs[2].plot(x,mr_new[2][-1],'-o',label='prediction')
# axxs[2].plot(x,mr_old[2][-1],'k+',label='ground truth')
# axxs[2].set_xlabel('\(x\)')
# axxs[2].set_ylabel('\(E\)')
# axxs[2].legend()
# ###tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/Hy_Intt.tex"))

for idx, del_var in enumerate(del_vars):
    axxxs[0,idx].plot(x,e_hynew[idx],'-o',label='prediction')
    axxxs[0,idx].plot(x,e_hyold[idx],'k+',label='ground truth')
    axxxs[0,idx].set_xlabel('\(x\)')
    axxxs[0,idx].set_ylabel('\(E\)')
    axxxs[0,idx].legend()
    axxxs[1,idx].plot(x,e_rarenew[idx],'-o',label='prediction')
    axxxs[1,idx].plot(x,e_rareold[idx],'k+',label='ground truth')
    axxxs[1,idx].set_xlabel('\(x\)')
    axxxs[1,idx].set_ylabel('\(E\)')
    axxxs[1,idx].legend()
####tikzplotlib.save(join(home,"rom-using-autoencoders/01_Thesis/Figures/Chapter_5/all_inttest.tex"))

plt.show()
