import numpy as np

#POD class

# class evaluate():
#     def __init__(self, c=None, level=None):
#         self.c = c
#         if level == "hy":
#             self.level = 3
#         else: 
#             self.level = 5

#     def pod(self):

#         u, s, vh = np.linalg.svd(self.c,full_matrices=False) #s Singularvalues
#         S = np.diagflat(s)
#         xx = u[:,:self.level]@S[:self.level,:level]@vh[:self.level,:]
#         return xx



def pod(c, level):
    if level == "hy":
        level = 3
    else: 
        level = 5
    
    u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
    S = np.diagflat(s)
    xx = u[:,:level]@S[:level,:level]@vh[:level,:]
    l2 = np.linalg.norm((c - xx).flatten())/np.linalg.norm(c.flatten()) # calculatre L2-Norm Error
    print(l2)
    return xx