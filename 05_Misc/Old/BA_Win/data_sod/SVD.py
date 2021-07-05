'''
SVD
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=17) 
plt.rc('ytick', labelsize=17) 

datalocation = ['sod25Kn0p01/f.mat','sod25Kn0p00001/f.mat','sod241Kn0p00001/f.mat']
q = 1 #Plot variable


for i in datalocation:
    
    #Load Data
    f = sio.loadmat(i)
    f = f['f']
    
    #Getting dimensions                                                     #t x v x x (25x40x200)
    shape = f.shape
    t = shape[0] 
    v = shape[1] 
    x = shape[2] 
    #Submatrix
    c = np.zeros((v,t*x))
    n = 0
    
    #Build 2D-Version
    for i in range(t):                                             # T (zeilen)
        for j in range(v):                                         # V (spalten)
                c[j,n:n+x]=f[i,j,:]
    
        n = n + x

    #SVD
    print(c.shape)
    u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
    
    x = np.linspace(-10,10,v) #Zeitvariable
    

    #Plot sigma over k Kn 0.01 (25 snaps)
    if q == 1:
        #Sigulärwerte
        k = len(s)
        k = range(k)
        plt.figure(1)
        plt.subplot(3,2,1)
        plt.semilogy(k,s,'.-')
        plt.legend(['Kn 0.01 25 snapshots'])
        plt.ylabel(r'$\sigma$',fontsize=17)
        #Plot Cumultative Energy over k
        plt.subplot(3,2,2)
        plt.plot(k,np.cumsum(s)/np.sum(s),'.-')
        plt.legend(['Kn 0.01 25 snapshots'])
        plt.ylabel('Cumultative Energy', fontsize=17)
        plt.figure(2)
        plt.subplot(3,2,1)
        plt.plot(x,u[:,0])
        plt.ylabel('U1',fontsize=17)
        plt.subplot(3,2,2)
        plt.plot(x,u[:,1])
        plt.ylabel('U2',fontsize=17)
        plt.subplot(3,2,3)
        plt.plot(x,u[:,2])
        plt.ylabel('U3',fontsize=17)
        plt.subplot(3,2,4)
        plt.plot(x,u[:,3])
        plt.ylabel('U4',fontsize=17)
        plt.subplot(3,2,5)
        plt.plot(x,u[:,4])
        plt.ylabel('U5',fontsize=17)
        plt.xlabel('v', fontsize=17)
        plt.subplot(3,2,6)
        plt.plot(x,u[:,5])
        plt.ylabel('U6',fontsize=17)
        plt.xlabel('v', fontsize=17)
    #Plot sigma over k KN 0.00001 (25 snaps)
    if q == 2:
        k = len(s)
        k = range(k)
        plt.figure(1)
        plt.subplot(3,2,3)
        plt.semilogy(k,s,'.-')
        plt.legend(['Kn 0.00001 25 snapshots'])
        plt.ylabel(r'$\sigma$',fontsize=17)
        #Plot Cumultative Energy over k
        plt.subplot(3,2,4)
        plt.plot(k,np.cumsum(s)/np.sum(s),'.-')
        plt.legend(['Kn 0.00001 25 snapshots'])
        plt.ylabel('Cumulative Energy', fontsize=17)
        plt.figure(3)
        plt.subplot(3,2,1)
        plt.plot(x,u[:,0])
        plt.ylabel('U1',fontsize=17)
        plt.subplot(3,2,2)
        plt.plot(x,u[:,1])
        plt.ylabel('U2',fontsize=17)
        plt.subplot(3,2,3)
        plt.plot(x,u[:,2])
        plt.ylabel('U3',fontsize=17)
        plt.subplot(3,2,4)
        plt.plot(x,u[:,3])
        plt.ylabel('U4',fontsize=17)
        plt.subplot(3,2,5)
        plt.plot(x,u[:,4])
        plt.ylabel('U5',fontsize=17)
        plt.xlabel('v', fontsize=17)
        plt.subplot(3,2,6)
        plt.plot(x,u[:,5])
        plt.ylabel('U6',fontsize=17)
        plt.xlabel('v', fontsize=17)
    if q == 3:
            #Plot sigma over k KN 0.00001 (241 snaps)
        k1 = len(s)    
        k1 = range(k1)
        plt.figure(1)
        plt.subplot(3,2,5)
        plt.semilogy(k1,s,'.-')
        plt.legend(['Kn 0.00001 241 snapshots'])
        plt.ylabel(r'$\sigma$',fontsize=17)
        plt.xlabel('k', fontsize=17)
        #Plot Cumultative Energy over k
        plt.subplot(3,2,6)
        plt.plot(k1,np.cumsum(s)/np.sum(s),'.-')
        plt.legend(['Kn 0.00001 241 snapshots'])
        plt.xlabel('k', fontsize=17)
        plt.ylabel('Cumulative Energy', fontsize=17)
        plt.figure(4)
        plt.subplot(3,2,1)
        plt.plot(x,u[:,0])
        plt.ylabel('U1',fontsize=17)
        plt.subplot(3,2,2)
        plt.plot(x,u[:,1])
        plt.ylabel('U2',fontsize=17)
        plt.subplot(3,2,3)
        plt.plot(x,u[:,2])
        plt.ylabel('U3',fontsize=17)
        plt.subplot(3,2,4)
        plt.plot(x,u[:,3])
        plt.ylabel('U4',fontsize=17)
        plt.subplot(3,2,5)
        plt.plot(x,u[:,4])
        plt.ylabel('U5',fontsize=17)
        plt.xlabel('v', fontsize=17)
        plt.subplot(3,2,6)
        plt.plot(x,u[:,5])
        plt.ylabel('U6',fontsize=17)
        plt.xlabel('v', fontsize=17)
        plt.show()
    q = q+1  
# Integration




