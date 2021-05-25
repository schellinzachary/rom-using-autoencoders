'''
Data-Preprocessing Convolutional
'''
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

from pathlib import Path
from os.path import join
home = str(Path.home())


rar = sio.loadmat(join(home,'rom-using-autoencoders/02_data_sod/sod25Kn0p01/f.mat'))
hy = sio.loadmat(join(home,'rom-using-autoencoders/02_data_sod/sod25Kn0p00001/f.mat'))

rar  = rar['f']
hy = hy['f'] 

#Use to produce training data for both flows together
flow = np.concatenate((hy,rar),axis=1)
flow = np.swapaxes(flow,0,1)
plt.imshow(flow[0,:,:])
plt.show()
np.random.shuffle(flow)

plt.imshow(flow[0,:,:])
plt.show()
flow = np.expand_dims(flow, axis=1)
exit()
np.save('Data/flow_4D.npy',flow)


#Use to produce inference datase for hydro and rare flow
# rar = np.swapaxes(rar,0,1)
# hy = np.swapaxes(hy,0,1)
# np.random.seed()
# np.random.shuffle(rar)
# np.random.shuffle(hy)
# rar = np.expand_dims(rar, axis=1)
# hy = np.expand_dims(hy, axis=1)
# np.save('Data/sod25Kn0p01_4D_unshuffled.npy',rar)
# np.save('Data/sod25Kn0p00001_4D_unshuffled.npy',hy)

