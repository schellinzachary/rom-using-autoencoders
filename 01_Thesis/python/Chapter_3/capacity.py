import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
###import tikzplotlib

home = str(Path.home())

x=np.linspace(0,10,num=60)
tl = np.exp(-x) + 1/x
vl=np.zeros(x.shape)
vl[:14] = np.exp(-x[:14]+2) + 1/x[:14]


vl[14:60] = 0.35 + np.exp(-x[14:60]+2)  + 0.008*x[14:60]**2.8
plt.plot(x,tl,'-+',label="Train Error")
plt.plot(x,vl,'-o',label="Val. Error")
plt.xlabel("Capacity")
plt.ylabel("Error")
plt.title("Example of Generalization Gap")
plt.legend()
#tikzplotlib.save('/home/zachi/capacity.tex')
plt.show()


