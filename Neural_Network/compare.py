import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# samples = np.load('preprocessed_samples_lin.npy')

# row_mean = np.mean(samples,axis=1)

# de_meaned = samples.copy()
# for i in range(4999):
# 	de_meaned[i] = de_meaned[i] - row_mean[i]

# print(de_meaned.mean(axis=1))

# plt.plot(de_meaned[2001])
# plt.show()

# def compare(samples):
# 	a = 0
# 	c = 0
# 	for i in range(4999):
# 		for j in range(4999):
# 			b = np.sum(np.abs(samples[i]-samples[j+1]))

# 			if b < 1e-6:
# 				a = a+1
# 				print(a)
# 				# if a < 500:
# 				# 	c = c+1
# 				# 	print(a,i)
# 	return a,i,c
# a, i, c = compare(de_meaned)
# print('a=',a,'i=',i, 'c=',c)

f = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/v.mat')
f  = f['v']

f = np.array(f)
print(f[2] - f[1])
