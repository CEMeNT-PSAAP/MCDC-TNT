"""
Name: Advance
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

from math import log
import numpy as np
import numba as nb


@nb.jit(nopython=True,) 
def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L):
    
    end_trans = 0
    #mesh_dist_traveled, mesh_dist_traveled_squared
    end_trans_vec = np.zeros(num_part, np.int32)
    
    while end_trans == 0:
        pre_p_mesh_cell = p_mesh_cell
        
        
        pre_end_trans_vec = end_trans_vec
        
        [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_time, p_dist, end_trans_vec] = AdvanceParticleRun(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx,
                                    p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
                                    mesh_total_xsec, end_trans_vec, L, num_part)
        
        for i in range(num_part):
            if (pre_end_trans_vec[i] == 0):
                mesh_dist_traveled[pre_p_mesh_cell[i]] += p_dist[i]
                mesh_dist_traveled_squared[pre_p_mesh_cell[i]] += p_dist[i]**2
                
        if np.sum(end_trans_vec) == 0:
            end_trans = 1
    
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared)

@nb.jit(nopython=True, parallel=True) 
def AdvanceParticleRun(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx,
                                    p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
                                    mesh_total_xsec, end_trans_vec, L, num_part):
    p_dist = np.zeros(num_part)
    for i in nb.prange(num_part):
            [p_pos_x[i], p_pos_y[i], p_pos_z[i], p_mesh_cell[i], p_time[i],
            p_dist[i], end_trans_vec[i]] = AdvanceParticle(p_pos_x[i], p_pos_y[i], p_pos_z[i], p_mesh_cell[i], dx,
                                    p_dir_y[i], p_dir_z[i], p_dir_x[i], p_speed[i], p_time[i],
                                    mesh_total_xsec, end_trans_vec[i], L)
                                    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_time, p_dist, end_trans_vec)


@nb.jit(nopython=True) 
def AdvanceParticle(x, y, z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
                    mesh_total_xsec, end_trans, L):
    
    kicker = 1e-10
    flag = 1
    dist_acum = 0
    if (end_trans == False):
        if (x < 0): #exited rhs
            end_trans = True
        elif (x >= L): #exited lhs
            flag = True
        else:
            dist = -log(np.random.random()) / mesh_total_xsec[p_mesh_cell]
            
            x_loc = (p_dir_x * dist) + x
            LB = p_mesh_cell * dx
            RB = LB + dx
            #print(type(x_loc))
            #print(type(LB))
            if (x_loc < LB):        #move partilce into cell at left
                dist_traveled = (LB - x)/p_dir_x + kicker
                cell_next = p_mesh_cell -1
            elif (x_loc > RB):      #move particle into cell at right
                dist_traveled = (RB - x)/p_dir_x + kicker
                cell_next = p_mesh_cell +1
            else:                   #move particle in cell
                dist_traveled = dist
                flag = True
                cell_next = p_mesh_cell
                
            x += p_dir_x*dist_traveled
            y += p_dir_y*dist_traveled
            z += p_dir_z*dist_traveled
            
            dist_acum += dist
            #mesh_dist_traveled[p_mesh_cell] += dist_traveled
            #mesh_dist_traveled_squared[p_mesh_cell] += dist_traveled**2
            
            p_mesh_cell = cell_next
            
            #advance particle clock
            p_time  += dist_traveled/p_speed

    return(x, y, z, p_mesh_cell, p_time, dist_acum, end_trans)


def StillIn(p_pos_x, surface_distances, p_alive, num_part):
    tally_left = 0
    tally_right = 0
    for i in range(num_part):
        #exit at left
        if p_pos_x[i] <= surface_distances[0]:
            tally_left += 1
            p_alive[i] = False
            
        elif p_pos_x[i] >= surface_distances[len(surface_distances)-1]:
            tally_right += 1
            p_alive[i] = False
            
    return(p_alive, tally_left, tally_right)
    
if __name__ == "__main__":
    n = int(1e8)
    n_m = int(50)
    L = 2
    dx = L/n_m
    
    p_pos_x = np.random.random(n)
    p_pos_y = np.random.random(n)  
    p_pos_z = np.random.random(n)  
    
    p_dir_x = np.random.random(n)  
    p_dir_y = np.random.random(n)
    p_dir_z = np.random.random(n)
    
    p_mesh_cell = np.random.randint(n_m, size=n)
    p_speed = np.ones(n, dtype=float)
    p_time = np.zeros(n, dtype=float)
    
    mesh_total_xsec = 1.0
    mesh_dist_traveled = np.zeros(n_m, dtype=float)
    mesh_dist_traveled_squared = np.zeros(n_m, dtype=float)
    
    Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            n, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L)
    
    print(mesh_dist_traveled)
    print(mesh_dist_traveled_squared)
    
