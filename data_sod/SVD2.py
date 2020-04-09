'''
SVD
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
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

    c = np.reshape(f,(v,t*x), order='F')

    u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
    
    print(s[0])
    print(len(s))

    #Plot sigma over k Kn 0.01 (25 snaps)
    if q == 1:
        k = len(s)
        k = range(k)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.subplot(3,2,1) 
        plt.plot(k,s,'.-')
        plt.legend(['Kn 0.01 25 snapshots'])
        plt.ylabel(r'$\sigma$',fontsize=17)
        #Plot Cumultative Energy over k
        plt.subplot(3,2,2)
        plt.plot(k,np.cumsum(s)/np.sum(s),'.-')
        plt.legend(['Kn 0.01 25 snapshots'])
        plt.ylabel('Cumultative Energy', fontsize=17)
        plt.rc('xtick', labelsize=20) 
        plt.rc('ytick', labelsize=20) 
        x = np.linspace(-10,10,40)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(x,u[:,0],'.-')
        plt.ylabel('$U_{1}$', fontsize=20)
        plt.show()
        plt.plot(x,u[:,1],'.-')
        plt.ylabel('$U_{2}$', fontsize=20)
        plt.show()
        plt.plot(x,u[:,2],'.-')
        plt.ylabel('$U_{3}$', fontsize=20)
        plt.show()
    #Plot sigma over k KN 0.00001 (25 snaps)
    if q == 2:
        k = len(s)
        k = range(k)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.subplot(3,2,3)
        plt.plot(k,s,'.-')
        plt.legend(['Kn 0.00001 25 snapshots'])
        plt.ylabel(r'$\sigma$',fontsize=17)
        #Plot Cumultative Energy over k
        plt.subplot(3,2,4)
        plt.plot(k,np.cumsum(s)/np.sum(s),'.-')
        plt.legend(['Kn 0.00001 25 snapshots'])
        plt.ylabel('Cumulative Energy', fontsize=17)
    if q == 3:
            #Plot sigma over k KN 0.00001 (241 snaps)
        k1 = len(s)    
        k1 = range(k1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.subplot(3,2,5)
        plt.plot(k1,s,'.-')
        plt.legend(['Kn 0.00001 241 snapshots'])
        plt.ylabel(r'$\sigma$',fontsize=17)
        plt.xlabel('k', fontsize=17)
        #Plot Cumultative Energy over k
        plt.subplot(3,2,6)
        plt.plot(k1,np.cumsum(s)/np.sum(s),'.-')
        plt.legend(['Kn 0.00001 241 snapshots'])
        plt.xlabel('k', fontsize=17)
        plt.ylabel('Cumulative Energy', fontsize=17)
        plt.show()

    q = q+1  
