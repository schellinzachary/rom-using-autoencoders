
'''
Density
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('xtick', labelsize=17) 
#plt.rc('ytick', labelsize=17) 



datalocation = ['sod25Kn0p01/f.mat','sod25Kn0p00001/f.mat','sod241Kn0p00001/f.mat']
datalocation2 = ['/home/zachary/Desktop/BA/data_sod/sod25Kn0p01auto/f.mat',
                '/home/zachary/Desktop/BA/data_sod/sod25Kn0p00001auto/f.mat',
                '/home/zachary/Desktop/BA/data_sod/sod241Kn0p00001auto/f.mat']
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

    t = shape[0] 
    v = shape[1] 
    x = shape[2] 
    #Submatrix
    c = np.zeros((v,t*x))
    print(c.shape)
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
        #ax1.plot(xi,rho3)
    if q == 2:
        #plt.subplot(3,1,2)
        ax2.plot(xi,rho)
        ax2.plot(xi,rho2,linestyle='dashdot')
        #ax2.plot(xi,rho3)
    if q == 3:
        #plt.subplot(3,1,3)
        ax3.plot(xi,rho)
        ax3.plot(xi,rho2,linestyle='dashdot')
        #ax3.plot(xi,rho3)
        plt.xlabel('x', fontsize='17')
        plt.ylabel('Density', fontsize='17')
    q = q+1
plt.legend()
plt.show()

print(c.shape)
print(xx.shape)
# #Visualizing

def density(c,predict):

    rho_predict = np.zeros([25,200])
    rho_samples = np.zeros([25,200])
    n=0

    for k in range(25):
        for i in range(200):
            rho_samples[k,i] = np.sum(c[i+n]) * 0.5128
            rho_predict[k,i] = np.sum(predict[i+n]) * 0.5128   
        n += 200
    return rho_samples, rho_predict

rho , rho2 = density(xx,c)

def visualize(c,predict):
    fig = plt.figure()
    ax = plt.axes(ylim=(0,1),xlim=(0,200))

    line1, = ax.plot([],[],label='original')
    line2, = ax.plot([],[],label='prediction')

    def init():
        line1.set_data([],[])
        line2.set_data([],[])
        return line1, line2


    def animate(i):
        print(i)
        line1.set_data(np.arange(200),c[i])
        line2.set_data(np.arange(200),predict[i])
        return line1, line2

    anim = animation.FuncAnimation(
                                   fig, 
                                   animate, 
                                   init_func = init,
                                   frames = 200,
                                   interval = 200,
                                   blit = True
                                   )

    ax.legend()
    plt.show()

visualize(rho,rho2)

print(np.sum(np.abs(rho - rho2)))