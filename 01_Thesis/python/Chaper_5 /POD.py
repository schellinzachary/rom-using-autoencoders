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

