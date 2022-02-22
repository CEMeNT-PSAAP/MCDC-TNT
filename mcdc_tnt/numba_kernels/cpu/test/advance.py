import numpy.random as rand
import numpy as np
from numba import jit
import numba
from timeit import default_timer as timer
import time


#@jit(nopython=True)
def advance (x, y, z, i_h, j_h, k_h, xi):
            
    mesh_total_xsec: float = 3.0154
    
    dist: float = -np.log(xi) / mesh_total_xsec
    
    x += i_h*dist
    y += j_h*dist
    z += k_h*dist
    
    return(x, y, z)
    
#@jit(nopython=True, parallel=True)
def advance_run(n, x, y, z, i_h, j_h, k_h, xi):
    for i in numba.prange(n):
        [x[i], y[i], z[i]] = advance(x[i], y[i], z[i], i_h[i], j_h[i], k_h[i], xi[i])
    return (x, y, z)
    
    
n = int(1e8)
x = 2.31*np.ones(n)
y = 2.31*np.ones(n)
z = 2.31*np.ones(n)

i_h = rand.random(n)
j_h = rand.random(n)
k_h = rand.random(n)

xi = rand.random(n)

print("Random Numbers Allocated")

start = timer()
[x,y,z] = advance_run(n, x, y, z, i_h, j_h, k_h, xi)
end = timer()

print(end - start)






