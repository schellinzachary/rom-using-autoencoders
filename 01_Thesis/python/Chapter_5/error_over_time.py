import numpy as np
import scipy.io as sio
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
##import tikzplotlib
from matplotlib import cm

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

def shapeback_field(c):  #Shape the reconstruction from 5000x40 bach to 25x40x200
    t = int(c.shape[0]/200)
    f = np.empty([t,40,200])
    n = 0
    for i in range(t):
        for j in range(200):
            f[i,:,j] = c[j+n,:]
        n += 200
    return(f)

def l2_time(org, rec):
    return(norm((rec - org),axis=(1,2)) / norm(org,axis=(1,2)))

fig,axs = plt.subplots(1,2)
fig.suptitle("Error over time for Hy (left) and Rare (right)")
figg,axxs = plt.subplots(2,4)
figg.suptitle("\tilde(f) for POD,the FCNNs, and CNN and f x =[0.375,0.75] and t=0.12; Hy top, Rare bottom")
im = ["im1","im2"]

for idx, level in enumerate(["hy", "rare"]):
    #POD
    method = "POD"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from POD import pod
    rec_pod, z = pod(c,level)

    rec_pod = shapeback_field(rec_pod)
    c = shapeback_field(c)
    err_pod = l2_time(c, rec_pod)

    method = "Fully"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from FullyConnected import fully
    rec_fully, z = fully(c, level)
    rec_fully = rec_fully.detach().numpy()
    rec_fully = shapeback_field(rec_fully)
    c = shapeback_field(c)
    err_fully = l2_time(c, rec_fully)
 
    method = "Conv"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from Conv import conv
    rec_conv, z = conv(c)
    rec_conv = rec_conv.detach().numpy()
    rec_conv = rec_conv.squeeze()
    c = c.squeeze()
    err_conv = norm((rec_conv - c),axis=(0,2)) / norm(c,axis=(0,2))

    t = np.linspace(start=0,stop=0.12,num=25,endpoint=True)
    x = np.linspace(start=0,stop=0.995,num=200)
    v = np.linspace(start=-10,stop=10,num=40)

    axs[idx].plot(t,err_pod,'k''-x',label="POD")
    axs[idx].plot(t,err_fully,'r''-o',label="Fully")
    axs[idx].plot(t,err_conv,'g''-v',label="Conv")
    axs[idx].set_xlabel("t")
    axs[idx].set_ylabel("L2-error")
    axs[idx].legend()

    im = axxs[idx,0].imshow(c[:,-1,75:150],
        cmap='gray',label="FOM",
        extent=[x[75],x[150],v[0],v[-1]],
        aspect="auto",
        origin="lower"
        )
    axxs[idx,0].set_xlabel("\(x\)")
    axxs[idx,0].set_ylabel("\(v\)")
    axxs[idx,1].imshow(rec_pod[-1,:,75:150],
        cmap='gray',label="POD",
        vmin=np.min(c),
        vmax=np.max(c),
        extent=[x[75],x[150],v[0],v[-1]],
        aspect="auto",
        origin="lower"
        )
    axxs[idx,1].set_xlabel("\(x\)")
    axxs[idx,1].set_ylabel("\(v\)")
    axxs[idx,2].imshow(rec_fully[-1,:,75:150],
        'gray',label="Fully",
        vmin=np.min(c),
        vmax=np.max(c),
        extent=[x[75],x[150],v[0],v[-1]],
        aspect="auto",
        origin="lower"
        )
    axxs[idx,3].imshow(rec_conv[:,-1,75:150],
        'gray',label="Conv",
        vmin=np.min(c),
        vmax=np.max(c),
        extent=[x[75],x[150],v[0],v[-1]],
        aspect="auto",
        origin="lower"
        )
    axxs[idx,3].set_xlabel("\(x\)")
    axxs[idx,3].set_ylabel("\(v\)")

    figg.colorbar(im, ax=axxs[idx])
##tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_5/ErrTime_test.tex'))
##tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_5/ErrWorst_test.tex'))
plt.show()