'''
Data-Preprocessing Convolutional
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np


data_set = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/f.mat')


dataset  = data_set['f']
#dataset = np.array(dataset,order='C')
dataset = np.reshape(dataset,(40,25,200),order='C')


plt.imshow(dataset[1,:,:])
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()


#Normalizing the Input

dataset = (dataset - np.min(dataset)) / ( np.max(dataset) - np.min(dataset) )


#dataset = np.expand_dims(dataset, axis=1)

#np.save('preprocessed_samples_conv',dataset)





#First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( )

a = dataset[0]
im = plt.imshow(a)

def animate_func(i):

    im.set_array(dataset[i])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               #frames = nSeconds * fps,
                               #interval = 1000 / fps, # in ms
                               )


plt.show()

