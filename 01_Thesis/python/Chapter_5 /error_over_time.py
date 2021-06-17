import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
from mpl_toolkits.axes_grid1 import AxesGrid

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

def l2_time(org, rec):
    return(norm((rec - org),axis=(1,2)) / norm(org,axis=(1,2)))

#fig,axs = plt.subplots(1,2)
figg,axxs = plt.subplots(2,4)
im = ["im1","im2"]

for idx, level in enumerate(["rare", "hy"]):
    #POD
    method = "POD"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    rec_pod = pod(c,level)
    rec_pod = shapeback_field(rec_pod)
    c = shapeback_field(c)
    err_pod = l2_time(c, rec_pod)

    method = "Fully"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from FullyConnected import fully
    rec_fully = fully(c, level)
    rec_fully = rec_fully.detach().numpy()
    rec_fully = shapeback_field(rec_fully)
    c = shapeback_field(c)
    err_fully = l2_time(c, rec_fully)
 
    method = "Conv"
    c = load_BGKandMethod(method, level) # load FOM data for evaluation
    from Conv import conv
    rec_conv = conv(c)
    rec_conv = rec_conv.detach().numpy()
    rec_conv = rec_conv.squeeze()
    c = c.squeeze()
    err_conv = norm((rec_conv - c),axis=(0,2)) / norm(c,axis=(0,2))

    t = np.linspace(start=0,stop=0.12,num=25,endpoint=True)
    # axs[idx].plot(t,err_pod,'k''-x',label="POD")
    # axs[idx].plot(t,err_fully,'r''-o',label="Fully")
    # axs[idx].plot(t,err_conv,'g''-v',label="Conv")
    # axs[idx].legend()
    for n_bin, ax in zip(n_bins, axxs.ravel()):
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
        # Fewer bins will result in "coarser" colomap interpolation
        im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cm)
        ax.set_title("N bins: %s" % n_bin)
        fig.colorbar(im, ax=ax) 
    axxs[idx,0].imshow(c[:,-1,75:150],'gray',label="FOM")
    axxs[idx,1].imshow(rec_pod[-1,:,75:150],'gray',label="POD")
    axxs[idx,2].imshow(rec_fully[-1,:,75:150],'gray',label="Fully")
    im[idx] = axxs[idx,3].imshow(rec_conv[:,-1,75:150],'gray',label="Conv")

cax,kw = matplotlib.colorbar.make_axes([ax for ax in axxs.flat])
plt.colorbar(im[idx], cax=cax, **kw)


##tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_5/ErrTime_test.tex'))
##tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_5/ErrWorst_test.tex'))
plt.show()