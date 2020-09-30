'''
SVD PLOTS for BA
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

fontsize=25
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize) 


#Load Data
f = sio.loadmat('sod25Kn0p01/f.mat')
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

u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues

S = np.diagflat(s)
xx = u[:,:5]@S[:5,:5]@vh[:5,:]
def plot_cumu():


	#Plot Cumultative Energy and Singular Vaules
	k = range(len(s))
	plt.figure(1)
	plt.subplot(1,2,1)
	plt.semilogy(k,s,'.-''k')
	plt.ylabel(r'$\sigma$',fontsize=fontsize)
	plt.xlabel(r'k',fontsize=fontsize)
	plt.subplot(1,2,2)
	plt.plot(k,np.cumsum(s)/np.sum(s),'.-''k')
	plt.ylabel('Cumultative Energy', fontsize=fontsize)
	plt.xlabel(r'k',fontsize=fontsize)
	plt.show()

	return

print('Test Error:',np.sum(np.abs(xx-c))/len(c))

# Plot the Density
def density_svd(c):

    rho_svd = np.zeros([25,200])
    n=0

    for k in range(25):
        for i in range(200):
            rho_svd[k,i] = np.sum(c[:,i+n]) * 0.5128
   
        n += 200
    return rho_svd


rho_svd = density_svd(xx)
rho = density_svd(c)

plt.plot(rho_svd[20],'-.''k')
plt.plot(rho[20],'-*''k')
plt.ylabel(r'$Density \quad \rho$',fontsize=fontsize)
plt.xlabel(r'$x$',fontsize=fontsize)
plt.show()

print(np.sum(np.abs(rho_svd - rho)))

### Overall mistakes sample-wise

mistake_list = []
for i in range(4999):
    mistake = np.sum(np.abs(xx[:,i] - c[:,i]))
    mistake_list.append((i,mistake))

zip(mistake_list)

plt.bar(range(len(mistake_list)),[val[1]for val in mistake_list],color='k')
plt.xlabel(r'$Samples$',fontsize=fontsize)
plt.ylabel(r'$Absolute Error$',fontsize=fontsize)
plt.grid()
plt.tight_layout()
plt.show()

plot_cumu()
# plt.figure(2)
# plt.subplot(3,2,1)
# plt.plot(x,u[:,0])
# plt.ylabel('U1',fontsize=17)
# plt.xlabel('v', fontsize=17)
# plt.subplot(3,2,2)
# plt.plot(x,u[:,1])
# plt.ylabel('U2',fontsize=17)
# plt.xlabel('v', fontsize=17)
# plt.subplot(3,2,3)
# plt.plot(x,u[:,2])
# plt.ylabel('U3',fontsize=17)
# plt.xlabel('v', fontsize=17)
# plt.subplot(3,2,4)
# plt.plot(x,u[:,3])
# plt.ylabel('U4',fontsize=17)
# plt.xlabel('v', fontsize=17)
# plt.subplot(3,2,5)
# plt.plot(x,u[:,4])
# plt.ylabel('U5',fontsize=17)
# plt.xlabel('v', fontsize=17)
# plt.subplot(3,2,6)
# plt.plot(x,u[:,5])
# plt.ylabel('U6',fontsize=17)
# plt.xlabel('v', fontsize=17)