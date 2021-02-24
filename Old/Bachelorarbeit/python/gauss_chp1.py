'''
BGK Model Macroscopic quantities
'''
from scipy import signal
import numpy as np
from numpy import linspace, zeros, sqrt, pi
import matplotlib.pyplot as plt
import tikzplotlib


#Variables
sig = 0.2
x = np.linspace(-0.5,1,num = 100)
t = np.linspace(0,0.5,num=100)
mu = 0.25



def gaussian(x, mu, sig):
    return (1 / (np.sqrt(2 * np.pi * sig**2)))*np.exp(-(x - mu)**2 / (2 * sig**2))

plt.plot(x,gaussian(x, mu, sig),'k',label = 'f')
plt.fill(x,gaussian(x, mu, sig),'dimgrey',label = 'rho')
plt.plot(t,np.linspace(0.9,0.9,100),'--''k')
plt.xlabel('v')
plt.ylabel('f')
plt.legend()
tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/Bachelorarbeit/Figures/BGK/Boltzmann.tex')
plt.show()





