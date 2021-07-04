"Plot figures"

import importlib.util
from os.path import join
from pathlib import Path
home = Path.home()

fig1 = "rom-using-autoencoders/01_Thesis/python/Chapter_2/gauss_example.py"
exec(open(join(home,fig1)).read())
fig2 = "rom-using-autoencoders/01_Thesis/python/Chapter_3/capacity.py"
exec(open(join(home,fig2)).read())
fig3 = "rom-using-autoencoders/01_Thesis/python/Chapter_4/cum_sum.py"
exec(open(join(home,fig3)).read())




fig4 = "rom-using-autoencoders/01_Thesis/python/Chapter_4/intrinsic_variables.py"
exec(open(join(home,fig4)).read())



import os 
os.system('python .py')