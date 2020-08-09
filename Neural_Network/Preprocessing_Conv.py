'''
Data-Preprocessing
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np


data_set = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/f.mat')


dataset  = data_set['f']
print(dataset.shape)

dataset = np.reshape(dataset,(40,25,200),order='A')


for i in range(40):
    dataset[i] = dataset[i]/np.linalg.norm(dataset)


for i in range(40):
    dataset[i] = dataset[i] - np.mean(dataset[i], axis=1,keepdims=True)



np.random.shuffle(dataset)

dataset = np.expand_dims(dataset, axis=1)

np.save('preprocessed_samples_conv',dataset)


# fps = 30
# nSeconds = 5


# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure( )

# a = dataset[0]
# im = plt.imshow(a)

# def animate_func(i):
#     if i % fps == 0:
#         print( '.', end ='' )

#     im.set_array(dataset[i])
#     return [im]

# anim = animation.FuncAnimation(
#                                fig, 
#                                animate_func, 
#                                frames = nSeconds * fps,
#                                interval = 1000 / fps, # in ms
#                                )


# plt.show()

