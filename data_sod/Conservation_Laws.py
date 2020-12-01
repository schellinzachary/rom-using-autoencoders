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

	m = f * v
	rho_u = np.sum(m,axis = 1) * dv
	u = rho_u / rho

	E = f * ((v**2) * .5)
	E = np.sum(E, axis = 1)


	T = ((2* E) / (3 * rho)) - (u**2 / 3)
	p = rho * T
	return(rho,p,u,E,T,m,rho_u)

rho, p, u, E, T, m, rho_u = macro(f,v)


E = np.sum(E, axis = 1)
d_dt = np.diff(E)/ np.mean(E)


plt.plot(d_dt,'-+''k')
plt.xlabel('t',fontsize=20)
plt.ylabel('(drho_u/dt)/t_mean',fontsize=20)
plt.show()