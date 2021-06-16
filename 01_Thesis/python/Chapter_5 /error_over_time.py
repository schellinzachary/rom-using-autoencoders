import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import tikzplotlib

from os.path import join
from pathlib import Path
home = Path.home()

from POD import pod

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

def shapeback_field(c):  #Shape the reconstruction from 5000x40 bach to 25x40x200
    t = int(c.shape[0]/200)
    f = np.empty([t,40,200])
    n = 0
    for i in range(t):
        for j in range(200):
            f[i,:,j] = c[j+n,:]
        n += 200
    return(f)

fig,axs = plt.subplots(1,2)
# figg,axxs = plt.subplots(2,4)

for idx, level in enumerate(["hy", "rare"]):
    #POD
    method = "POD"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    rec_pod = pod(c,level)
    rec_pod = shapeback_field(rec_pod)
    c = shapeback_field(c)
    err_pod = norm((rec_pod - c),axis =(1,2)) / norm(c,axis=(1,2))

    method = "Fully"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from FullyConnected import fully
    rec_fully = fully(c, level)
    rec_fully = rec_fully.detach().numpy()
    rec_fully = shapeback_field(rec_fully)
    c = shapeback_field(c)
    err_fully = norm((rec_fully - c),axis =(1,2)) / norm(c,axis=(1,2))

    method = "Conv"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from Conv import conv
    rec_conv = conv(c)
    rec_conv = rec_conv.detach().numpy()
    rec_conv = rec_conv.squeeze()
    c = c.squeeze()
    err_conv = norm((rec_conv - c),axis =(0,2)) / norm(c,axis=(0,2))

    t = np.linspace(start=0,stop=0.12,num=25,endpoint=True)
    axs[idx].plot(t,err_pod,'k''-x',label="POD")
    axs[idx].plot(t,err_fully,'r''-o',label="Fully")
    axs[idx].plot(t,err_conv,'g''-v',label="Conv")
    axs[idx].legend()
    # axxs[idx,0].imshow(c[:,-1,75:150],'gray',label="FOM")
    # axxs[idx,1].imshow(rec_pod[-1,:,75:150],'gray',label="POD")
    # axxs[idx,2].imshow(rec_fully[-1,:,75:150],'gray',label="Fully")
    # axxs[idx,3].imshow(rec_conv[:,-1,75:150],'gray',label="Conv")

###tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_5/ErrTime_test.tex'))
plt.show()