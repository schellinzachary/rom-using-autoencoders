'''
Data-Preprocessing Linear
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np
import sys




f = sio.loadmat('/home/zachi/Documents/ROM_using_Autoencoders/data_sod/sod25Kn0p00001/f.mat')
f  = f['f']

def crazyD(f):
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
  return(c)
def twoD(f):
  x=200
  t=25
  v=40

  #Submatrix
  c = np.empty((t*x,v))
  n = 0

  #Build 2D-Version
  for i in range(t): 
    print(n)                                            
    for j in range(x):
      c[j+n,:]=f[i,:,j]
    n +=200
        

  return(c)

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

def normalize(a):
  return (a - np.min(a)) / (np.max(a) - np.min(a))


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

c = twoD(f)

o = crazyD(f)

plt.plot(c[4998])
plt.plot(o[:,4999])
plt.show()

print(np.sum(np.abs(c))-np.sum(np.abs(f)))

#np.random.shuffle(c)
#
#np.save('Data/sod25Kn0p00001_2D.npy',c)




# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(-10,40),ylim=(0,1))

# line, = ax.plot([],[],label='i =')

# def init():
#   line.set_data([],[])
#   return line,


# def animate(i):
#   a = np.arange(40)
#   line.set_data(a,c[i+2000])
#   line.set_label('i = {}'.format(i))
#   return line,

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
