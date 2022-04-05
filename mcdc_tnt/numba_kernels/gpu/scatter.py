"""
Name: Scatter
breif: Adding fission particles to phase vectors for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np
from numba import cuda

@cuda.jit
def ScatterCuda(d_scatter_indices, p_dir_x, p_dir_y, p_dir_z, rands):
    
    i = cuda.grid(1)
    
    if (i < d_scatter_indices.size):

        # Sample polar and azimuthal angles uniformly
        mu  = 2.0*rands[2*i] - 1.0
        azi = 2.0*math.pi*rands[2*i+1]
        
        # Convert to Cartesian coordinate
        c = (1.0 - mu**2)**0.5
        p_dir_y[d_scatter_indices[i]] = math.cos(azi)*c
        p_dir_z[d_scatter_indices[i]] = math.sin(azi)*c
        p_dir_x[d_scatter_indices[i]] = mu


def Scatter(scatter_indices, scat_count, p_dir_x, p_dir_y, p_dir_z, rands):
    """
    NUMBA CUDA Kernel: Isotropically chosses new particle directions after a scatter event

    Parameters
    ----------
    scatter_indices : vector int
        Indicies to PSV of particls that will be undergoing transport.
    scat_count : int
        number of particles to scatter.
    p_dir_y : vector double
        PSV: y direction unit value of phase space particles (index is particle value).
    p_dir_z : vector double
         PSV: z direction unit value of phase space particles (index is particle value).
    p_dir_x : vector double
         PSV: x direction unit value of phase space particles (index is particle value).
    rands : vector doubles
        from an rng, length: 2*scat_count.

    Returns
    -------
    None.

    """
    d_scatter_indices = cuda.to_device(scatter_indices)
    d_p_dir_x = cuda.to_device(p_dir_x)
    d_p_dir_y = cuda.to_device(p_dir_y)
    d_p_dir_z = cuda.to_device(p_dir_z)
    d_p_rands = cuda.to_device(rands)
    
    threadsperblock = 32
    blockspergrid = (scat_count + (threadsperblock - 1)) // threadsperblock
    ScatterCuda[blockspergrid, threadsperblock](d_scatter_indices, d_p_dir_x, d_p_dir_y, d_p_dir_z, d_p_rands)
    
    p_dir_x = d_p_dir_x.copy_to_host()
    p_dir_y = d_p_dir_y.copy_to_host()
    p_dir_z = d_p_dir_z.copy_to_host()
    
    return(p_dir_x, p_dir_y, p_dir_z)
    

def test_Scatter():
    
    scat_count = 3
    scatter_indices = np.array([0,1,4], dtype=int)
    p_dir_x = np.array([1,2,0,0,4])
    p_dir_y = np.array([1,2,0,0,4])
    p_dir_z = np.array([1,2,0,0,4])
    rands = np.array([1,1,0,0,.5,.5])
    
    
    [p_dir_x, p_dir_y, p_dir_z] = Scatter(scatter_indices, scat_count, p_dir_x, p_dir_y, p_dir_z, rands)
    
    
    assert(p_dir_y[0] == 0)
    assert(p_dir_z[0] == 0)
    assert(p_dir_x[0] == 1)
    
    assert(p_dir_y[1] == 0)
    assert(p_dir_z[1] == 0)
    assert(p_dir_x[1] == -1)
    
    assert(p_dir_y[4] == -1)
    assert(np.allclose(p_dir_z[4], 0))
    assert(p_dir_x[4] == 0)
    
if __name__ == '__main__':
    test_Scatter()
