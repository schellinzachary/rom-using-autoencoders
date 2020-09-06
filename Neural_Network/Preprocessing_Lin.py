'''
Data-Preprocessing Linear
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


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
def meaned(c):
  row_mean = np.mean(c,axis=1)

  de_meaned = c.copy()
  for i in range(4999):
    de_meaned[i] = de_meaned[i] - row_mean[i]
  return de_meaned

def delete(c):
  n=0
  p=0
  g=np.zeros(1250)
  for j in range(25):
    for i in range(50):
      g[i+p] = i+n

    n += 200
    p += 50
  return np.delete(c,g,0)

g = delete(c)

def normalize(a):
  return (a - np.min(a)) / (np.max(a) - np.min(a))

g = normalize(g)


np.save('preprocessed_samples_lin_substract50_normalized',g)





# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(-10,40),ylim=(0,1))

# line, = ax.plot([],[],label='i =')

# def init():
# 	line.set_data([],[])
# 	return line,


# def animate(i):
# 	a = np.arange(40)
# 	line.set_data(a,c[i+2000])
# 	line.set_label('i = {}'.format(i))
# 	return line,

# anim = animation.FuncAnimation(
#                                fig, 
#                                animate, 
#                                init_func = init,
#                                frames = 200,
#                                interval = 20,
#                                blit = True
#                                )

# ax.legend()
# plt.show()

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
        line1.set_data(np.arange(150),c[i])
        line2.set_data(np.arange(150),predict[i])
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

def density(x):

  rho_samples = np.zeros([25,150])
  n=0

  for k in range(25):
      for i in range(150):
          rho_samples[k,i] = np.sum(x[i+n]) * 0.5128  
      n += 150
  return rho_samples


rho = density(g)
visualize(rho,rho)
