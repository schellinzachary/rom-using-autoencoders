'''
Data-Preprocessing Linear
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np


f = sio.loadmat('/home/zachary/Desktop/BA/data_sod/sod25Kn0p01/f.mat')
f  = f['f']


x=200
t=25
v=40

#Submatrix
c = np.zeros((t*x,v))
n = 0

#Build 2D-Version
for i in range(t):                                             # T (zeilen)
    for j in range(v):                                         # V (spalten)
            c[n:n+x,j]=f[i,j,:]

    n = n + x


np.save('preprocessed_samples_lin',c)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-10,40),ylim=(0,1))

line, = ax.plot([],[],label='i =')

def init():
	line.set_data([],[])
	return line,


def animate(i):
	a = np.arange(40)
	line.set_data(a,c[i+2000])
	line.set_label('i = {}'.format(i))
	return line,

anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               init_func = init,
                               frames = 200,
                               interval = 20,
                               blit = True
                               )

ax.legend()
plt.show()