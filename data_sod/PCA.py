'''
PCA
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=17) 
plt.rc('ytick', labelsize=17) 

for i in datalocation:
    
    #Load Data
    f = sio.loadmat(i)
    f = f['f']
    
    #Getting dimensions                                                     
    shape = f.shape
    t = shape[0] 
    v = shape[1] 
    x = shape[2] 
    #Submatrix & Build 2D-Version
    c = np.zeros((v,t*x))
    n = 0
    for i in range(t):                                             
        for j in range(v):                                         # V (spalten)
                c[j,n:n+x]=f[i,j,:]
    
        n = n + x


    #Mean over time at x = x_mid = x_100
    p = 100
    mean = np.zeros((v,t*x))
    for k in range(t):
        mean[:,p] = np.sum(c[:,p])*(1/v)
        p = p+x
    B = c - mean
    B = np.asmatrix(B)
    C = (1/(1-v))*(B.H)@B
    #Eigendecomposition
    w,v = np.linalg.eig(C)