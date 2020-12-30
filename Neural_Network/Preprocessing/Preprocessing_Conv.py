'''
Data-Preprocessing Convolutional
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np
import torch

def teashape(f):
	a = np.empty((40,25,200),order='C')
	for i in range(f.shape[1]):
		a[i] = f[:,i,:]
	return(a)




data_set = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p01/f.mat')


dataset  = data_set['f']

dataset = teashape(dataset)

#np.random.shuffle(dataset)

# plt.imshow(dataset[0])
# plt.xlabel('x')
# plt.ylabel('t')
# plt.colorbar()
# plt.show()

dataset = np.expand_dims(dataset, axis=1)

np.save('preprocessed_samples_conv_unshuffled',dataset)




#First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure( )

# a = dataset[0]
# im = plt.imshow(a)

# def animate_func(i):

#     im.set_array(dataset[i])
#     return [im]

# anim = animation.FuncAnimation(
#                                fig, 
#                                animate_func, 
#                                #frames = nSeconds * fps,
#                                #interval = 1000 / fps, # in ms
#                                )


# plt.show()
