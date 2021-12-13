#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: CleanUp
breif: Misc functions for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import numpy as np

def SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive,
        num_part, meshwise_fission_pdf, particle_speed, isotropic=True):
    """
    Parameters
    ----------
    particle phase space perameters:
        p_pos_x : vector(float)
        p_pos_y : vector(float)
        p_pos_z : vector(float)
        p_region : vector(int)
        p_dir_y : vector(float)
        p_dir_z : vector(float)
        p_dir_x : vector(float)
        p_speed : vector(float)
        p_time : vector(float)
        
        
    problem geometry perameters
        num_part : int
            How many particles are there.
        x_rhs_gen : float
            right hand limit of the generating region in slab geo.
        x_lhs_gen : float
            left limit of the generating region in slab.
        L_gen : float
            width of generating region.
        generation_region : int
            index of generating region.
        particle_speed : float
            particle speed.
        isotropic : Bool, optional
            is the source isotropic or uniform. The default is True.

    Returns
    -------
    All pahse space perameters.
    """
    
    for i in range(num_part):
        # Position
        
        #find mesh cell birth based on provided pdf
        xi = np.random.random()
        cell = 0
        summer = 0
        while (summer < xi):
            summer += meshwise_fission_pdf[cell]
            cell += 1
                
        cell -=1
        p_mesh_cell[i] = cell
        
        #sample birth location within cell
        p_pos_x[i] = dx*cell + dx*np.random.random()
        p_pos_y[i] = 0.0
        p_pos_z[i] = 0.0
        
        
        # Direction
        if isotropic:
            # Sample polar and azimuthal angles uniformly
            mu  = 2.0*np.random.random() - 1.0
            azi = 2.0*np.pi*np.random.random()
    	
            # Convert to Cartesian coordinate
            c = (1.0 - mu**2)**0.5
            p_dir_y[i] = np.cos(azi)*c
            p_dir_z[i] = np.sin(azi)*c
            p_dir_x[i] = mu
        else:
            p_dir_x[i] = 1.0
            p_dir_y[i] = 0.0
            p_dir_z[i] = 0.0
    
        # Speed
        p_speed[i] = particle_speed
    
        # Time
        p_time[i] = 0.0
        
        p_alive[i] = True
        
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive)


