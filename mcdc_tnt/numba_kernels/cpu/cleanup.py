"""
Name: CleanUp
breif: Misc functions for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""



def BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, p_event, num_part):
    kept = 0
    for i in range(num_part):
        if p_alive[i] == True:
        #     if p_mesh_cell[i] > 9:
        #         print("index from this round:")
        #         print(i)
        #         print("index for next round:")
        #         print(kept)
            
            p_pos_x[kept] = p_pos_x[i]
            p_pos_y[kept] = p_pos_y[i]
            p_pos_z[kept] = p_pos_z[i]
            
            # Direction
            p_dir_x[kept] = p_dir_x[i]
            p_dir_y[kept] = p_dir_y[i]
            p_dir_z[kept] = p_dir_z[i]
            
            # Speed
            p_speed[kept] = p_speed[i]
            
            # Time
            p_time[kept] = p_time[i]
            
            # Regions
            p_mesh_cell[kept] = p_mesh_cell[i]
            
            # Flags
            p_alive[kept] = p_alive[i] 
            p_event[kept] = p_event[i]
            kept +=1
            
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, kept)
