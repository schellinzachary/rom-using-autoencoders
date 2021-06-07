import numpy as np

#POD class
class pod(object):
    def load(c, level):
        u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
        S = np.diagflat(s)
        xx = u[:,:level]@S[:level,:level]@vh[:level,:]
        return xx, s


#Compute POD for intrinsic variables variation
def intr_eval(c,iv):
    rec, s = pod.load(c,iv)
    l2 = np.linalg.norm((c - rec).flatten())/np.linalg.norm(c.flatten()) # calculatre L2-Norm Error
    return(l2)
