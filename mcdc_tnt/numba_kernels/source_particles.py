"""
Name: CleanUp
breif: Misc functions for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive,
        num_parts, meshwise_fission_pdf, particle_speed, isotropic=True):
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
    
    for i in nb.prange(num_parts):
        # Position
        
        #find mesh cell birth based on provided pdf
        #xi = np.random.random()
        #cell = 0
        #summer = 0
        #while (summer < xi):
        #    summer += meshwise_fission_pdf[cell]
        #    cell += 1
                
        #cell -=1+
        cell = int(meshwise_fission_pdf.size/2)
        p_mesh_cell[i] = cell
        
        #sample birth location within cell
        p_pos_x[i] = cell*dx + dx/2
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




def test_SourceParticles():
    num_parts = 5
    p_pos_x = np.zeros(num_parts)
    p_pos_y = np.zeros(num_parts)
    p_pos_z = np.zeros(num_parts)
    
    p_mesh_cell = np.zeros(num_parts)
    
    p_dir_x = np.zeros(num_parts)
    p_dir_y = np.zeros(num_parts)
    p_dir_z = np.zeros(num_parts)
    
    p_speed = np.zeros(num_parts)
    p_time = np.ones(num_parts)
    p_alive = np.zeros(num_parts)
    
    particle_speed = 1
    meshwise_fission_pdf = [0,1]
    
    iso=False
    
    dx = 0.2
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive] = SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_parts, meshwise_fission_pdf, particle_speed, iso)
    
    assert (np.sum(p_time) == 0)
    assert (p_mesh_cell.all() == 1)
    assert (p_alive.all() == True)
    assert (p_pos_x.all() > .2)
    
if __name__ == '__main__':
    test_SourceParticles()    

