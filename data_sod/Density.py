
'''
Density
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=17) 
plt.rc('ytick', labelsize=17) 



datalocation = ['sod25Kn0p01/f.mat','sod25Kn0p00001/f.mat','sod241Kn0p00001/f.mat']
datalocation2 = ['A:/Desktop/BA/data_sod/sod25Kn0p01auto/f.mat','A:/Desktop/BA/data_sod/sod25Kn0p00001auto/f.mat','A:/Desktop/BA/data_sod/sod241Kn0p00001auto/f.mat']
q = 1 #Plot variable
fig, (ax1,ax2,ax3) = plt.subplots(3, sharex = True, sharey=True)

#for i in datalocation:
for i,b in zip( datalocation, datalocation2):
    
    #Load Data
    f = sio.loadmat(i)
    fa = sio.loadmat(b)
    f = f['f']
    fa = fa['fa']
    
    #Getting dimensions                                                     #t x v x x (25x40x200)
    shape = f.shape
    print(f.shape)
    print(fa.shape)
    t = shape[0] 
    v = shape[1] 
    x = shape[2] 
    #Submatrix
    c = np.zeros((v,t*x))
    n = 0
    
    #Build 2D-Version
    pl = 0
    for i in range(t):                                             # T (zeilen)
        for j in range(v):                                         # V (spalten)
                c[j,n:n+x]=f[i,j,:]
    
        n = n + x

    #SVD

    u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
    # Density
    S = np.diagflat(s)
    xx = u[:,:3]@S[:3,:3]@vh[:3,:]
    xi = np.linspace(0,1,x) #Weg 
    rho = np.zeros(x)       #Dichte initialisieren
    rho2 = np.zeros(x)
    rho3 = np.zeros(x)
    dimc = c.shape      
    enddimc = dimc[1]
    lasttime = enddimc - x  #Letzter Zeitschritt
    
    
    fa = fa.reshape(v,x*t)

    for p in range(x):
        rho[p] = np.sum(c[:,lasttime+p],axis=None)*0.5128
        rho2[p]= np.sum(xx[:,lasttime+p],axis=None)*0.5128
        rho3[p]= np.sum(fa[:,lasttime+p],axis=None)*0.5128
    if q == 1:
        #plt.subplot(3,1,1)
        ax1.plot(xi,rho,label='hallo')
        ax1.plot(xi,rho2,linestyle='dashdot')
        ax1.plot(xi,rho3)
    if q == 2:
        #plt.subplot(3,1,2)
        ax2.plot(xi,rho)
        ax2.plot(xi,rho2,linestyle='dashdot')
        ax2.plot(xi,rho3)
    if q == 3:
        #plt.subplot(3,1,3)
        ax3.plot(xi,rho)
        ax3.plot(xi,rho2,linestyle='dashdot')
        ax3.plot(xi,rho3)
        plt.xlabel('x', fontsize='17')
        plt.ylabel('Density', fontsize='17')
    q = q+1
plt.show()