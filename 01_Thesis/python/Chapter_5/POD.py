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

def count_parameters(u,S,vh):
    return(u.size+S.size+vh.size)

def pod(c, level):
    if level == "hy":
        level = 3
    else: 
        level = 5
    
    u, s, vh = np.linalg.svd(c,full_matrices=False) #s Singularvalues
    S = np.diagflat(s)
    xx = u[:,:level]@S[:level,:level]@vh[:level,:]

    paramcount = count_parameters(u[:,:level],S[:level,:level],vh[:level,:])
    #print(paramcount)
    return xx, u[:,:level]