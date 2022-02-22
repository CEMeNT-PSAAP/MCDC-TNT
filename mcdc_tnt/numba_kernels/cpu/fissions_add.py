"""
Name: FissionsAdd
breif: Adding fission particles to phase vectors for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 18th 2021
"""
import numpy as np

def FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive,
                fis_count, nu_new_neutrons, fission_event_index, num_part, particle_speed, rands):
    
    k=0 #index for fission temp vectors
    for i in range(fis_count):
        for j in range(nu_new_neutrons):
            # Position
            p_pos_x[k+num_part] = p_pos_x[fission_event_index[i]]
            p_mesh_cell[k+num_part] = p_mesh_cell[fission_event_index[i]]
            p_pos_y[k+num_part] = p_pos_y[fission_event_index[i]]
            p_pos_z[k+num_part] = p_pos_x[fission_event_index[i]]
            
            # print("fission particle produced")
            # print("from particle {0} and indexed as particle {1}".format(fission_event_index[i], k+num_part))
            # print("produced at: {0}".format(p_pos_x[k+num_part]))
            # Direction
            # Sample polar and azimuthal angles uniformly
            mu  = 2.0*rands[i+j] - 1.0
            azi = 2.0*rands[i+j+1]
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
