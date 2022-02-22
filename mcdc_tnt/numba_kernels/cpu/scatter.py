"""
Name: Scatter
breif: Adding fission particles to phase vectors for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import numpy as np

def Scatter(scatter_indices, scat_count, p_dir_x, p_dir_y, p_dir_z, rands):

    for i in range(scat_count):

            # Sample polar and azimuthal angles uniformly
            mu  = 2.0*rands[i] - 1.0
            azi = 2.0*np.pi*rands[i+1]
    	
            # Convert to Cartesian coordinate
            c = (1.0 - mu**2)**0.5
            p_dir_y[scatter_indices[i]] = np.cos(azi)*c
            p_dir_z[scatter_indices[i]] = np.sin(azi)*c
            p_dir_x[scatter_indices[i]] = mu
            
    return(p_dir_x, p_dir_y, p_dir_z)
