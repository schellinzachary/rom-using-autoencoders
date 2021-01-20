#testing if its possible to get density relation different to total relation and yes its possible

import numpy as np

a = np.random.rand(2,10)


b = np.random.rand(2,10)

c = np.random.rand(2,10)

d = np.random.rand(2,10)


print(np.sum(np.abs(a -b)))
print(np.sum(np.abs(c -d)))

def dense(x):

    dense = np.zeros([5,2])
    n=0

    for k in range(5):
        for i in range(2):
            dense[k,i] = np.sum(x[:,i+n]) * 0.5128
   
        n += 2
    return dense

d_a = dense(a)

d_b = dense(b)

d_c = dense(c)

d_b= dense(b)


print(np.sum(np.abs(d_a-d_b)))

print(np.sum(np.abs(d_c - d_b)))

