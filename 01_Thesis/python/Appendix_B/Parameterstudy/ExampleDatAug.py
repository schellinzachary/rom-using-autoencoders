'''
Example data augmentation
'''

from pathlib import Path
from os.path import join
home = str(Path.home())
loc_data = "rom-using-autoencoders/04_Autoencoder/Preprocessing/Data/flow_4D.npy"
loc_plot = "rom-using-autoencoders/01_Thesis/Figures/Parameterstudy/Convolutional/ExpDatAug3.tex"
loc_chpt = "rom-using-autoencoders/01_Thesis/python/Appendix_B/Parameterstudy"

import matplotlib.pyplot as plt
##import tikzplotlib

import numpy as np
import torch
import torch.tensor as tensor
from torchvision.transforms.functional import rotate, vflip

#load & scale data
f = np.load(join(home,loc_data))
f = tensor(f, dtype=torch.float)

#Augment data with flipping and rotating
f_rotate = rotate(f,angle=180)
f_flip = vflip(f)
#f = torch.cat((f,f_rotate,f_flip))

fig, ax = plt.subplots(1,3)

ax[0].imshow(f[0,0,:,:].squeeze(),cmap='gray',origin='lower')
ax[1].imshow(f_rotate[0,0,:,:].squeeze(),cmap='gray',origin='lower')
im = ax[2].imshow(f_flip[0,0,:,:].squeeze(),cmap='gray',origin='lower')
fig.colorbar(im)
fig.suptitle("Examples of dataaugmentation for CNN")
###tikzplotlib.save(join(home,loc_plot))
plt.show()