import numpy as np


class pod(object):
    def __init__(self,c,level):
        self.level = level
        self.c = c
    def load(c, level):
        if level == "hy":
            level = 3
        else:
            level = 5
        print(level)
        u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
        S = np.diagflat(s)
        xx = u[:,:level]@S[:level,:level]@vh[:level,:]
        return xx, s


#Compute POD for intrinsic variables variation
###############################################
def intr_eval(c,iv):
    rec, s = pod.load(c,iv)
    l2 = np.linalg.norm((c - rec).flatten())/np.linalg.norm(c.flatten()) # calculatre L2-Norm Error
    return(l2)