'''
Consrvation Laws
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np
import sys



f = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/f.mat')
v = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/v.mat')
t = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/t.mat')
t  = t['treport']
f  = f['f']
v  = v['v']

t.squeeze()
t=t.T


def macro(f,v):
	dv = v[1]- v[0]
	rho = np.sum(f,axis =1) * dv

	rho_u = np.sum(f * (v) ,axis = 1) * dv
	u = rho_u / rho

	E = f * ((v**2) * .5)
	E = np.sum(E, axis = 1)


	T = ((2* E) / (3 * rho)) - (u**2 / 3)
	p = rho * T
	return(rho,p,u,E,T,rho_u)

rho, p, u, E, T, rho_u = macro(f,v)



d_dt = np.diff(np.sum(rho_u, axis =1))/ (np.sum(rho_u, axis = (0,1)))


plt.plot(d_dt,'-+''k')
plt.xlabel('t',fontsize=20)
plt.ylabel('(dp/dt)/t_mean',fontsize=20)
# plt.ylim(0,1e-2)
plt.show()