"""
Name: FissionsAdd
breif: Adding fission particles to phase vectors for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 18th 2021
"""
import numpy as np
import numba as nb


@nb.jit(nopython=True)
def FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive,
                fis_count, nu_new_neutrons, fission_event_index, num_part, particle_speed, rands):
    """
    Run advance for a

    Parameters
    ----------
    p_pos_x : vector double
        PSV: x position of phase space particles (index is particle value).
    p_pos_y : vector double
        PSV: y position of phase space particles (index is particle value).
    p_pos_z : vector double
        PSV: z position of phase space particles (index is particle value).
    p_mesh_cell : vector int
        PSV: mesh cell location of a given particle.
    p_dir_y : vector double
        PSV: y direction unit value of phase space particles (index is particle value).
    p_dir_z : vector double
         PSV: z direction unit value of phase space particles (index is particle value).
    p_dir_x : vector double
         PSV: x direction unit value of phase space particles (index is particle value).
    p_speed : vector double
        PSV: speed (energy) or a particle (index is particle).
    p_time : vector double
        PSV: particle clock.
    p_alive : vector bool
        PSV: is it alive?
    fis_count : int
        how many fissions where recorded in smaple event.
    nu_new_neutrons : int
        how many neutrons produced per fission.
    fission_event_index : vector int
        indicies of particles that underwent fission after sample event.
    num_part : int
        number of particles currently under transport (indxed form 1).
    particle_speed : double
        speed of fissioned particles.
    rands : vector double
        produced from an rng, needs to be fis_count*nu*2.

    Returns
    -------
    Phase space variables with new fissions added.

    """
    k=0 #index for fission temp vectors
    for i in range(fis_count):
        for j in range(nu_new_neutrons):
            # Position
            p_pos_x[k+num_part] = p_pos_x[fission_event_index[i]]
            p_mesh_cell[k+num_part] = p_mesh_cell[fission_event_index[i]]
            p_pos_y[k+num_part] = p_pos_y[fission_event_index[i]]
            p_pos_z[k+num_part] = p_pos_z[fission_event_index[i]]
            
            # print("fission particle produced")
            # print("from particle {0} and indexed as particle {1}".format(fission_event_index[i], k+num_part))
            # print("produced at: {0}".format(p_pos_x[k+num_part]))
            # Direction
            # Sample polar and azimuthal angles uniformly
            mu  = 2.0*rands[4*i+2*j] - 1.0
            azi = 2.0*rands[4*i+2*j+1]
            # Convert to Cartesian coordinate
            c = (1.0 - mu**2)**0.5
            p_dir_y[k+num_part] = np.cos(azi)*c
            p_dir_z[k+num_part] = np.sin(azi)*c
            p_dir_x[k+num_part] = mu
                  
            # Speed
            p_speed[k+num_part] = particle_speed
            
            # Time
            p_time[k+num_part] = p_time[fission_event_index[i]]

            # Flags
            p_alive[k+num_part] = True
            
            k+=1
            
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, k)
    
    
    

def test_FissionsAdd():
    
    L = 1
    dx = .25
    N_m = 4
    
    num_part = 3
    p_pos_x = np.array([.55, 3, 5])
    p_pos_y = np.array([10, 3, 5])
    p_pos_z = np.array([15, 3, 5])
    
    p_mesh_cell = np.array([2, 87, -1])
    
    p_dir_x = np.ones(num_part)
    p_dir_y = np.zeros(num_part)
    p_dir_z = np.zeros(num_part)
    
    p_speed = np.ones(num_part)
    p_time = np.zeros(num_part)
    p_alive = np.ones(num_part, bool)
    p_alive[0] = False
    
    fis_count = 1
    nu = 2
    fission_event_index = [0]
    
    rands = [1,1,1,1]
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, k] = FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, fis_count, nu, fission_event_index, 1, 1, rands)
    
    print(p_pos_x)
    print(p_pos_y)
    print(p_pos_z)
    
    assert(np.allclose(p_pos_x, [0.55, 0.55, 0.55]))
    assert(np.allclose(p_pos_y, [10,10,10]))
    assert(np.allclose(p_pos_z, [15,15,15]))
    assert(p_dir_x.all() == 1)
    assert(p_alive[1:2].all() == True)
    
    
if __name__ == '__main__':
    test_FissionsAdd()
    
