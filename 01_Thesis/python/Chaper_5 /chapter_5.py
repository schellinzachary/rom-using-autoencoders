### Producing the plots for for Chapter 5
### 
### Usage : 1. Choose a method: Fully, Conv
###         2. Choose rarefaction level: hy, rare
### Author : Zachary
### Date   : 23.01.21
########################################

#import nececccasry libraries
import scipy.io as sio
import numpy as np
from numpy.linalg import norm 
import matplotlib.pyplot as plt
#import tikzplotlib
import torch
import torch.tensor as tensor
from scipy import interpolate



method = "Fully" # one of ["Fully" , "Conv" or "POD"]
level = "hy" # one of ["hy", "rare"]
iv=1

#A shape function
#################
def shapeback_field(c):
    t = int(c.shape[0]/200)
    f = np.empty([t,40,200])
    n = 0
    for i in range(t):
        for j in range(200):
            f[i,:,j] = c[j+n,:]
        n += 200
    return(f) #Shape the reconstruction from 5000x40 bach to 25x40x200

def shapeback_code(z):
    c = np.empty((25,z.shape[1],200))
    n=0
    for i in range(25):
        for p in range(200):
          c[i,:,p] = z[p+n,:].detach().numpy()
        n += 200
    return(c) # shaping back the code

def shape_AE_code(g):
    
    c = np.empty((g.shape[0]*g.shape[2],g.shape[1]))
    for i in range(g.shape[1]):
        n = 0
        for t in range(g.shape[0]):
          c[n:n+200,i] = g[t,i,:]
          n += 200
    return(c)

#load the full order BGK data
#############################

def load_BGKandMethod():
    if (method == 'Fully' or method=="POD") and level == 'hy' and train == 'No':
        c = np.load('Data/sod25Kn0p00001_2D_unshuffled.npy')
    elif (method == 'Fully' or method=="POD") and level == 'hy' and train == 'Yes':
        c = np.load('Data/sod25Kn0p00001_2D.npy')
    elif (method == 'Fully' or method=="POD") and level == 'rare' and train == 'No':
        c = np.load('Data/sod25Kn0p01_2D_unshuffled.npy')
    elif (method == 'Fully' or method=="POD") and level == 'rare' and train == 'Yes':
        c = np.load('Data/sod25Kn0p01_2D.npy')
    elif method == 'Conv' and level == 'hy' and train == 'No':
        c = np.load('Data/sod25Kn0p00001_4D_unshuffled.npy')
    elif method == 'Conv' and level == 'hy' and train == 'Yes':
        c = np.load('Data/sod25Kn0p00001_4D.npy')
    elif method == 'Conv' and level == 'rare' and train == 'No':
        c = np.load('Data/sod25Kn0p01_4D_unshuffled.npy')   
    else:
        c = np.load('Data/sod25Kn0p01_4D.npy')

    print("Method:",method,"Level:",level, "Train:",train)
    v = sio.loadmat('Data/sod25Kn0p01/v.mat')
    t = sio.loadmat('Data/sod25Kn0p01/t.mat')
    x = sio.loadmat('Data/sod25Kn0p01/x.mat')
    x = x['x']
    v = v['v']
    t  = t['treport']
    x = x.squeeze()
    t=t.squeeze()
    t=t.T

    return x,v,t,c

#load the full oder BGK data for hy and 251 snapshots
def load_BGK_241():
    c = np.load('Data/sod241Kn0p00001_2D_unshuffled.npy')
    return(c)





#evaluate the models
####################

train = "No"    # We don't need to train, the models are already trained
x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
if method == "Fully":
  from FullyConnected import model # import the method
  c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
  rec, code = model(c)
  c = c.detach().numpy()
  rec = rec.detach().numpy()
elif method == "Conv":
  from Convolutional import model
  c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
  rec, code = model(c)
  c = c.detach().numpy()
  rec = rec.detach().numpy()
else:
  from POD import pod as model
  rec, code = model(c)
        
l2 = np.linalg.norm((c - rec).flatten())/np.linalg.norm(c.flatten()) # calculatre L2-Norm Error
print('L2-Norm Error =',l2)

#Variation of intrinsic variables
#################################
# train = "Yes"   # We want to train the models again and change the int. vars.
# int_vars = [1,2,4,8,16,32] # number of intrinsic varibales to check
# study = "intvar"
# for level in ["rare", "hy"]:
#   for iv in int_vars:
          # method = "Conv"
          # x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
          # from Convolutional import model_train
          # c = tensor(c,dtype=torch.float)
          # model_train(c,iv,level)

            # method = "Fully"
            # x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
            # from FullyConnected import model_train
            # c = tensor(c,dtype=torch.float)
            # model_train(c,iv,level,study)



#Let the FCNN train with a different number of snapshots
######################################################
# train = "Yes"
# study = "snapshot"
# method = "hy"
# c = load_BGK_241()
# from FullyConnected import model_train
# c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
# model_train(c,3,level,study)



#Plot the results of the variation of intrinsic variables
#########################################################

# train = "No"
# fig,axs = plt.subplots(1,2)
# i = 0
# for level in ["hy", "rare"]:
#     pod = []
#     fully = []
#     conv = []
#     for iv in [1,2,4,8,16,32]:
        
#         #For POD
#         ########
#         method = "POD"
#         x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#         from POD import intr_eval
#         l2_pod = intr_eval(c,iv)
#         pod.append(l2_pod)
#         #For Fully
#         ##########
#         method = "Fully"
#         x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#         from FullyConnected import intr_eval
#         c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
#         l2_fully = intr_eval(c,iv,level)
#         fully.append(l2_fully)
#         #For Conv
#         #########
#         method = "Conv"
#         x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#         from Convolutional import intr_eval
#         c = tensor(c,dtype=torch.float)
#         l2_conv = intr_eval(c,iv,level)
#         conv.append(l2_conv)
        
#     axs[i].semilogy([1,2,4,8,16,32],pod,'k''x',label="POD")
#     axs[i].semilogy([1,2,4,8,16,32],fully,'r''o',label="Fully")
#     axs[i].semilogy([1,2,4,8,16,32],conv,'g''v',label="Conv")
#     axs[i].grid(True,which="both")
#     print(level)
#     axs[i].legend()
#     i+=1
# #tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/01_Thesis/Figures/Results/Var_iv.tex')
# plt.show()



#Plot mistakes over time and worst mistakes
###########################################

# train = "No"
# fig,axs = plt.subplots(1,2)
# figg,axxs = plt.subplots(2,4)
# i=0
# for level in ["hy", "rare"]:
#     #For POD
#     ########
#     method = "POD"
#     x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#     from POD import pod as model
#     rec, code = model(c)
#     rec_pod = shapeback_field(rec)
#     c = shapeback_field(c)
#     err_pod = norm((rec_pod - c),axis =(1,2)) / norm(c,axis=(1,2))
#     #For Fully
#     ##########
#     method = "Fully"
#     x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#     from FullyConnected import model
#     c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
#     rec, code = model(c)
#     c = c.detach().numpy()
#     rec = rec.detach().numpy()
#     rec_fully = shapeback_field(rec)
#     c = shapeback_field(c)
#     err_fully = norm((rec_fully - c),axis =(1,2)) / norm(c,axis=(1,2))
#     #For Conv
#     #########
#     method = "Conv"
#     x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#     from Convolutional import model
#     c = tensor(c,dtype=torch.float)
#     rec, code = model(c)
#     c = c.detach().numpy()
#     rec = rec.detach().numpy()
#     rec_conv = rec.squeeze()
#     c = c.squeeze()
#     err_conv = norm((rec_conv - c),axis =(0,2)) / norm(c,axis=(0,2))

#     axs[i].plot(t,err_pod,'k''-x',label="POD")
#     axs[i].plot(t,err_fully,'r''-o',label="Fully")
#     axs[i].plot(t,err_conv,'g''-v',label="Conv")
#     axs[i].legend()
#     axxs[i,0].imshow(c[:,-1,75:150],'gray',label="FOM")
#     axxs[i,1].imshow(rec_pod[-1,:,75:150],'gray',label="POD")
#     axxs[i,2].imshow(rec_fully[-1,:,75:150],'gray',label="Fully")
#     axxs[i,3].imshow(rec_conv[:,-1,75:150],'gray',label="Conv")
#     i+=1
# #tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/01_Thesis/Figures/Results/ErrWorst.tex')
# plt.show()

#Calculate macroscopic quantities from FOM and reconstructions and plot them
############################################################################

def macro(f,v):
    dv = v[1]- v[0]
    rho = np.sum(f,axis = 1) * dv
    rhou = f * v
    rhou = np.sum((rhou),axis = 1) * dv
    E = f * ((v**2) * .5) * dv
    E = np.sum(E, axis = 1)
    return(rho,rhou,E)
def conservation(rho,rhou,E):
    dtrho = np.gradient(np.sum(rho,axis=1))
    dtrhou = np.gradient(np.sum(rhou,axis=1))
    dtE = np.gradient(np.sum(E,axis=1))
    return(dtrho,dtrhou,dtE)




# train="No"
# fig,ax = plt.subplots(2,3) # for macroscopic quantities
# figg,axxs = plt.subplots(2,3) # for conservation
# i=0
# for level in ["hy","rare"]:
#     #For POD
#     ########
#     method = "POD"
#     x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#     from POD import pod as model
#     rec, code = model(c)
#     rec_pod = shapeback_field(rec)
#     c = shapeback_field(c)
#     rho_pod,rhou_pod,e_pod = macro(rec_pod,v)
#     #calculate the conservation
#     dtrho_pod,dtrhou_pod,dte_pod = conservation(rho_pod,rhou_pod,e_pod)
#     #For Fully
#     ##########
#     method = "Fully"
#     x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#     from FullyConnected import model
#     c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
#     rec, code = model(c)
#     c = c.detach().numpy()
#     rec = rec.detach().numpy()
#     rec_fully = shapeback_field(rec)
#     c = shapeback_field(c)
#     rho_fully,rhou_fully,e_fully = macro(rec_fully,v)
#     dtrho_fully,dtrhou_fully,dte_fully = conservation(rho_fully,rhou_fully,e_fully)
#     #For Conv
#     #########
#     method = "Conv"
#     x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
#     from Convolutional import model
#     c = tensor(c,dtype=torch.float)
#     rec, code = model(c)
#     c = c.detach().numpy()
#     rec = rec.detach().numpy()
#     rec_conv = np.swapaxes(rec.squeeze(),0,1)
#     c = np.swapaxes(c.squeeze(),0,1)
#     rho_conv,rhou_conv,e_conv = macro(rec_conv,v)
#     rho_fom,rhou_fom,e_fom = macro(c,v)
#     dtrho_fom,dtrhou_fom,dte_fom = conservation(rho_fom,rhou_fom,e_fom)
#     dtrho_conv,dtrhou_conv,dte_conv = conservation(rho_conv,rhou_conv,e_conv)



#     ax[i,0].plot(x,rho_fom[-1],'k''-x',label='FOM',markevery=5,markersize=5)
#     ax[i,0].plot(x,rho_pod[-1],'r''-o',label='POD',markevery=5,markersize=5)
#     ax[i,0].plot(x,rho_fully[-1],'p''--',label='FCNN',markevery=5,markersize=5)
#     ax[i,0].plot(x,rho_conv[-1],'g''v',label='CNN',markevery=5,markersize=5)
#     ax[i,0].set_ylabel('rho')
#     ax[i,0].set_xlabel('x')
#     ax[i,0].legend()
#     ax[i,1].plot(x,rhou_fom[-1],'k''-x',label='FOM',markevery=5,markersize=5)
#     ax[i,1].plot(x,rhou_pod[-1],'r''-o',label='POD',markevery=5,markersize=5)
#     ax[i,1].plot(x,rhou_fully[-1],'p''--',label='FCNN',markevery=5,markersize=5)
#     ax[i,1].plot(x,rhou_conv[-1],'g''v',label='CNN',markevery=5,markersize=5)
#     ax[i,1].set_ylabel('rho u')
#     ax[i,1].set_xlabel('x')
#     ax[i,1].legend()
#     ax[i,2].plot(x,e_fom[-1],'k''-x',label='FOM',markevery=5,markersize=5)
#     ax[i,2].plot(x,e_pod[-1],'r''-o',label='POD',markevery=5,markersize=5)
#     ax[i,2].plot(x,e_fully[-1],'p''--',label='FCNN',markevery=5,markersize=5)
#     ax[i,2].plot(x,e_conv[-1],'g''v',label='CNN',markevery=5,markersize=5)
#     ax[i,2].set_ylabel('E')
#     ax[i,2].set_xlabel('x')
#     ax[i,2].legend()

#     axxs[i,0].plot(t,dtrho_fom,'k''-x',label='FOM',markevery=5,markersize=5)
#     axxs[i,0].plot(t,dtrho_pod,'r''-o',label='POD',markevery=5,markersize=5)
#     axxs[i,0].plot(t,dtrho_fully,'p''--',label='FCNN',markevery=5,markersize=5)
#     axxs[i,0].plot(t,dtrho_conv,'g''v',label='CNN',markevery=5,markersize=5)
#     axxs[i,0].set_ylabel('rho')
#     axxs[i,0].set_xlabel('t')
#     axxs[i,0].legend()
#     axxs[i,1].plot(t,dtrhou_fom,'k''-x',label='FOM',markevery=5,markersize=5)
#     axxs[i,1].plot(t,dtrhou_pod,'r''-o',label='POD',markevery=5,markersize=5)
#     axxs[i,1].plot(t,dtrhou_fully,'p''--',label='FCNN',markevery=5,markersize=5)
#     axxs[i,1].plot(t,dtrhou_conv,'g''v',label='CNN',markevery=5,markersize=5)
#     axxs[i,1].set_ylabel('rho u')
#     axxs[i,1].set_xlabel('t')
#     axxs[i,1].legend()
#     axxs[i,2].plot(t,dte_fom,'k''-x',label='FOM',markevery=5,markersize=5)
#     axxs[i,2].plot(t,dte_pod,'r''-o',label='POD',markevery=5,markersize=5)
#     axxs[i,2].plot(t,dte_fully,'p''--',label='FCNN',markevery=5,markersize=5)
#     axxs[i,2].plot(t,dte_conv,'g''v',label='CNN',markevery=5,markersize=5)
#     axxs[i,2].set_ylabel('E')
#     axxs[i,2].set_xlabel('t')
#     axxs[i,2].legend()
#     i+=1
#     ######tikzplotlib.save('/home/zachi/ROM_using_Autoencoders/01_Thesis/Figures/Results/Conservation.tex')###
# plt.show()
    

#Interpolate in time with the FCNN
##################################

train="No"
method="Fully"
level="hy"
x,v,t,c = load_BGKandMethod() # load FOM data for evaluation
from FullyConnected import model
c = tensor(c,dtype=torch.float)  # make input data "c" a tensor
rec, code = model(c)
code = shapeback_code(code)
cnew=np.empty([241,3,200])
tnew=np.linspace(0.0,0.12,num=241)
for i in range(3):
    f = interpolate.interp1d(t[::1],code[::1,i,:],axis=0,kind='quadratic')
    cnew[:,i,:]=f(tnew)
#print(np.sum(np.abs(cnew)-np.abs(code)))



codenew = shape_AE_code(cnew)
codenew=tensor(codenew,dtype=torch.float)
from FullyConnected import decoder
fnew = decoder(codenew)
fnew=fnew.detach().numpy()
fnew = shapeback_field(fnew)
#fold = c.detach().numpy()
fold = np.load('Data/sod241Kn0p00001_2D_unshuffled.npy')
fold = shapeback_field(fold)
l2 = np.linalg.norm((fnew - fold).flatten())/np.linalg.norm(fold.flatten()) # calculatre L2-Norm Error
print(l2)
rho_old,rhou_old,e_old = macro(fold,v)
rho_new,rhou_new,e_new = macro(fnew,v)
print(rho_new.shape)
plt.plot(rhou_new[-1],'-o',label='prediction')
plt.plot(rhou_old[-1],'k+',label='ground truth')
plt.legend()

plt.show()





















