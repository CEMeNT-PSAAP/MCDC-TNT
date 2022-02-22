#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:52:42 2022

@author: jack
"""

import numpy.random as rand
import numpy as np
from numba import jit
import numba
from timeit import default_timer as timer
import time

n = int(1e8)
rand.seed(777)

#@jit(nopython=True)
def scatter(i_hat, j_hat, k_hat, rand1, rand2):
    mu = 2.0*rand1 - 1.0
    azi = 2.0*rand2 - 1.0
    
    c = (1.0 - mu**2)**0.5
    
    i_hat = np.cos(azi) * c
    j_hat = np.sin(azi) * c
    k_hat = mu
    
    return(i_hat, j_hat, k_hat)

#@jit(nopython=True, parallel=True)
def run_scatter(n, p_dir_x, p_dir_y, p_dir_z, xi):
    for i in numba.prange(n):
        [p_dir_x[i], p_dir_y[i], p_dir_z[i]] = scatter(p_dir_x[i], p_dir_y[i], p_dir_z[i], xi[2*i],xi[2*i+1])
        
    return(p_dir_x, p_dir_y, p_dir_z)



#def cuda_scatter(n, p_dir_x, p_dir_y, p_dir_z):
#    tx = cuda.threadIdx.x
    # Block id in a 1D grid
#    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
#    bw = cuda.blockDim.x
#    pos = tx + ty * bw


p_dir_y = rand.random(n)# create a 1D
p_dir_z = rand.random(n)
p_dir_x = rand.random(n)
xi = rand.random(2*n)

#timer = pk.Timer()

print("Random Numbers Allocated")

start = timer()
[p_dir_x, p_dir_y, p_dir_z] = run_scatter(n, p_dir_x, p_dir_y, p_dir_z, xi)
end = timer()

print(end - start)



