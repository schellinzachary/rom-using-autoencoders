'''
SVD PLOTS for BA
'''

import scipy.io as sio
import numpy as np
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
from matplotlib import rc

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':15})

# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


#Load Data

c = np.load('/home/zachi/Documents/ROM_using_Autoencoders/Neural_Network/Preprocessing/Data/sod25Kn0p00001_2D_unshuffled.npy')
c=c.T


plt.plot(np.arange(-20,20),c[:,900])
plt.plot(np.arange(-20,20),c[:,700])
plt.plot(np.arange(-20,20),c[:,500])
plt.plot(np.arange(-20,20), c[:,4501])
plt.show()

#SVD

u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues

S = np.diagflat(s)
xx = u[:,:3]@S[:3,:3]@vh[:3,:]
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
plt.ylabel(r'$Density \quad \rho$',fontsize=25)
plt.xlabel(r'$x$',fontsize=25)
plt.show()

print('Density Error:',np.sum(np.abs(rho_svd - rho))/25)
print('Summed Eucledian Distances:',np.sum(np.abs(c - xx))/5000)
print(c.shape)
print(xx.shape)
### Overall mistakes sample-wise

mistake_list = []
for i in range(4999):
    mistake = np.sum(np.abs(xx[:,i] - c[:,i]))
    mistake_list.append((i,mistake))

zip(mistake_list)

# ax = plt.subplot(111, polar=False)
# bars = ax.bar(range(len(mistake_list)),[val[1]for val in mistake_list],color='k',width=1)
# axr = ax.twiny()    
# axr.xaxis.set_major_locator(plt.FixedLocator(np.arange(0,25)))
# axr.set_xlim((0,25))
# ax.set_xlim((0,4999))
# ax.yaxis.grid(True)
# axr.xaxis.grid(True)
# ax.set_xlabel(r'$Samples$')
# axr.set_xlabel(r'$Timesteps$')
# ax.set_ylabel(r'$Absolute Error$')
# plt.show()

#plot_cumu()
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

predict = xx
test_error = norm((c[:] - predict[:]).flatten())/norm(c[:].flatten())
print(test_error)
#test_error = np.sum(np.abs(c - predict),axis=0)
# mean = np.sum(test_error)/len(test_error)
# print('Mean Test Error', mean)
# print('STD Test Error', ((1/(len(test_error)-1)) * np.sum((test_error - mean)**2 )))
# print('Abweichung vom Mean',np.sum(np.abs(test_error - mean)) / len(test_error))
# print('Highest Sample Error',np.max(test_error))
# print('Lowest Sample Error', np.min(test_error))