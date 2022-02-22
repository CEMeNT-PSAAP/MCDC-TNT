"""
Name: Scatter
breif: Adding fission particles to phase vectors for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np

def Scatter(scatter_indices, scat_count, p_dir_x, p_dir_y, p_dir_z, rands):
    """
    Isotropically chosses new particle directions after a scatter event

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

    for i in range(scat_count):

        # Sample polar and azimuthal angles uniformly
        mu  = 2.0*rands[2*i] - 1.0
        azi = 2.0*math.pi*rands[2*i+1]
	    
        # Convert to Cartesian coordinate
        c = (1.0 - mu**2)**0.5
        p_dir_y[scatter_indices[i]] = math.cos(azi)*c
        p_dir_z[scatter_indices[i]] = math.sin(azi)*c
        p_dir_x[scatter_indices[i]] = mu
            
    return(p_dir_x, p_dir_y, p_dir_z)
    
def test_Scatter():
    
    scat_count = 3
    scatter_indices = [0,1,4]
    p_dir_x = [1,2,0,0,4]
    p_dir_y = [1,2,0,0,4]
    p_dir_z = [1,2,0,0,4]
    rands = [1,1,0,0,.5,.5]
    
    
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
