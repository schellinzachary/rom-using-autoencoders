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
	m = np.sum(m,axis = 1) * dv
	u = m / rho

	E = f * ((v**2) * .5)
	E = np.sum(E, axis = 1)

	T = ((2* E) / (3 * rho)) - (u**2 / 3)
	p = rho * T
	return(rho,p,u,E,T,m)

def d_dt(r,t):
	d_dt=np.empty([24])
	
	r = np.sum(r,axis=1)
	for i in range(24):
		dt = t[i+1] - t[i]
		print(dt)
		d_dt[i] = (r[i+1] - r[i]) / dt
		print((r[i+1] - r[i]) / dt)
	return(d_dt)

rho, p, u, E, T, m = macro(f,v)


d_dt_E = d_dt(E,t)

dddd = np.diff(np.sum(E,axis=1))

#plt.plot(np.sum(E,axis=1),'-+''k')
plt.plot(dddd/np.mean(t))
plt.xlabel('t')
plt.ylabel('dE/dt')
plt.show()