import numpy as np
from __main__ import level


if level == "hy":
    num_mod = 3
else:
    num_mod = 5

def pod(c):
    u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
    S = np.diagflat(s)

    xx = u[:,:num_mod]@S[:num_mod,:num_mod]@vh[:num_mod,:]
    return xx, s


#Compute POD for intrinsic variables variation
###############################################
def intr_eval(c,iv):
    rec,s = pod(c,iv)
    l2 = np.linalg.norm((c - rec).flatten())/np.linalg.norm(c.flatten()) # calculatre L2-Norm Error
    return(l2)
