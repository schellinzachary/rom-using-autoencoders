'''
Data-Preprocessing Linear
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np
import sys


#Load the FOM data you want to preprocess
#########################################

f = sio.loadmat('/home/zachi/ROM_using_Autoencoders/02_data_sod/sod241Kn0p00001/f.mat') 
f  = f['f']

def twoD(f):
  x=f.shape[2]
  t=f.shape[0]
  v=f.shape[1]

  #Submatrix
  c = np.empty((t*x,v))
  n = 0

  #Build 2D-Version
  for i in range(t):                                         
    for j in range(x):
      c[j+n,:]=f[i,:,j]
    n +=200
        

  return(c)

c = twoD(f)

print(np.sum(np.abs(c))-np.sum(np.abs(f))) #check the reshaping

# np.random.shuffle(c) #shuffle the set

# np.save('Data/sod241Kn0p00001_2D.npy',c)



