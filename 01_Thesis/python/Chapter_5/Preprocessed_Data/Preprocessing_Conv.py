'''
Data-Preprocessing Convolutional
'''
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


rar = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p01/f.mat')
hy = sio.loadmat('/home/zachi/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/f.mat')

rar  = rar['f']
hy = hy['f'] 

rar = np.swapaxes(rar,0,1)
hy = np.swapaxes(hy,0,1)



# np.random.seed()
# np.random.shuffle(rar)
# np.random.shuffle(hy)

# plt.imshow(dataset[0])
# plt.xlabel('x')
# plt.ylabel('t')
# plt.colorbar()
# plt.show()

rar = np.expand_dims(rar, axis=1)
hy = np.expand_dims(hy, axis=1)

np.save('Data/sod25Kn0p01_4D_unshuffled.npy',rar)
np.save('Data/sod25Kn0p00001_4D_unshuffled.npy',hy)

