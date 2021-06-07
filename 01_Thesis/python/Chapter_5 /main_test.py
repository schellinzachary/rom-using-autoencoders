#import nececccasry libraries
import scipy.io as sio
import numpy as np
from numpy.linalg import norm 
import matplotlib.pyplot as plt
import tikzplotlib
import torch
import torch.tensor as tensor
from scipy import interpolate
from test import modeli

c = np.load('Data/sod25Kn0p00001_4D.npy')
c = tensor(c,dtype=torch.float)
for level in ["hy","rare"]:
	#build model
	rec, code = modeli.load(level,c)
	print(code.shape)
