'''
Plot Learning Rate Analysis Results
'''

import os
import numpy as np
import matplotlib.pyplot as plt


plt.figure()
parent_dir = "/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Conv_Nets/Learning_rate"
b = np.empty([5,2])
for i in range(3,5):

    for j in range(5):
        directory = f"Learning_rate_{i}"
        path = os.path.join(parent_dir,directory)

        a = np.load(f'%s/Train_losses_lr{i}_{j}.npy'%path)

        plt.semilogy(np.linspace(1,6000,num=6000,endpoint=True),a[0])

plt.show()






