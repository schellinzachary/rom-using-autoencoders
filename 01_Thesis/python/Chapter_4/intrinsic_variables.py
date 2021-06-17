import numpy as np
import matplotlib.pyplot as plt
###import tikzplotlib

from os.path import join
from pathlib import Path
home = Path.home()


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


#Plot the results of the variation of intrinsic variables
fig,axs = plt.subplots(1,2)

for idx, level in enumerate(["hy","rare"]):
    pod = []
    fully = []
    conv = []
    for iv in [1,2,4,8,16,32]:

        #For POD
        c = load_BGKandMethod("POD",level) # load FOM data for evaluation
        from POD import intr_eval
        l2_pod = intr_eval(c,iv)
        pod.append(l2_pod)

        #For Conv
        c = load_BGKandMethod("Conv", level) # load FOM data for evaluation
        from Conv import intr_eval
        l2_conv = intr_eval(c,iv,level)
        conv.append(l2_conv)

        #For Fully
        c = load_BGKandMethod("Fully", level) # load FOM data for evaluation
        from FullyConnected import intr_eval
        l2_fully = intr_eval(c,iv,level)
        fully.append(l2_fully)

 
    axs[idx].semilogy([1,2,4,8,16,32],pod,'k''x',label="POD")
    axs[idx].semilogy([1,2,4,8,16,32],conv,'g''v',label="Conv")
    axs[idx].semilogy([1,2,4,8,16,32],fully,'r''o',label="Fully")
    axs[idx].semilogy([1,2,4,8,16,32],fully,'r''o',label="Fully")
    axs[idx].set_title("%s"%level)
    axs[idx].grid(True,which="both")
    axs[idx].legend()
###tikzplotlib.save(join(home,'rom-using-autoencoders/01_Thesis/Figures/Chapter_4/Var_iv.tex'))
plt.show()
