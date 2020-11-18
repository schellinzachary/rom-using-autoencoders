'''
Data-Preprocessing Convolutional
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np


data_set = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p01/f.mat')


dataset  = data_set['f']
print(dataset.T.shape)

dataset = np.reshape(dataset,(40,25,200),order='C')



#Normalizing the Input

#dataset = (dataset - np.min(dataset)) / ( np.max(dataset) - np.min(dataset) )


dataset = np.expand_dims(dataset, axis=1)
np.random.shuffle(dataset)


np.save('preprocessed_samples_conv',dataset)





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

