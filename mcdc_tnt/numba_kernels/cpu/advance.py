"""
Name: Advance
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np
import numba as nb

@nb.jit(nopython=True)
def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L):
    
    p_end_trans = np.zeros(num_part)
    end_flag = 0
    max_mesh_index = len(mesh_total_xsec)-1
    
    cycle_count = 0
    while end_flag == 0:
        #allocate randoms
        rands = np.random.rand(num_part)
        #vector of indicies for particle transport
        
        p_dist_travled = np.zeros(num_part)
        
        pre_p_mesh = p_mesh_cell
        
        Advance_launch_threads(p_pos_x, p_pos_y, p_pos_z,
                          p_dir_y, p_dir_z, p_dir_x, 
                          p_mesh_cell, p_speed, p_time,  
                          dx, mesh_total_xsec, L,
                          p_dist_travled, p_end_trans, rands, num_part)
        
        end_flag = 1
        for i in range(num_part):
            if (0 < pre_p_mesh[i] < max_mesh_index):
                mesh_dist_traveled[pre_p_mesh[i]] += p_dist_travled[i]
                mesh_dist_traveled_squared[pre_p_mesh[i]] += p_dist_travled[i]**2
                
            if p_end_trans[i] == 0:
                end_flag = 0
            
        summer = p_end_trans.sum()
        cycle_count += 1
    
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared)

@nb.jit(nopython=True, parallel=True) 
def Advance_launch_threads(p_pos_x, p_pos_y, p_pos_z,
                          p_dir_y, p_dir_z, p_dir_x, 
                          p_mesh_cell, p_speed, p_time,  
                          dx, mesh_total_xsec, L,
                          p_dist_travled, p_end_trans, rands, num_part):
                          
    for i in nb.prange(num_part):
        #print(type(i))
        [p_pos_x[i], p_pos_y[i], p_pos_z[i], p_mesh_cell[i], p_time[i], p_dist_travled[i], p_end_trans[i]] = Advance_cycle(
                      p_pos_x[i], p_pos_y[i], p_pos_z[i],
                      p_dir_y[i], p_dir_z[i], p_dir_x[i], 
                      p_mesh_cell[i], p_speed[i], p_time[i],  
                      dx, mesh_total_xsec, L,
                      p_dist_travled[i], p_end_trans[i], rands[i])



@nb.jit(nopython=True) 
def Advance_cycle(p_pos_x, p_pos_y, p_pos_z,
                  p_dir_y, p_dir_z, p_dir_x, 
                  p_mesh_cell, p_speed, p_time,  
                  dx, mesh_total_xsec, L,
                  p_dist_travled, p_end_trans, rands):

    kicker = 1e-10

    if (p_end_trans == 0):
        if (p_pos_x < 0): #exited rhs
            p_end_trans = 1
        elif (p_pos_x >= L): #exited lhs
            p_end_trans = 1
            
        else:
            dist = -math.log(rands) / mesh_total_xsec[p_mesh_cell]
            
            x_loc = (p_dir_x * dist) + p_pos_x
            LB = p_mesh_cell * dx
            RB = LB + dx
            
            if (x_loc < LB):        #move partilce into cell at left
                p_dist_travled = (LB - p_pos_x)/p_dir_x + kicker
                cell_next = p_mesh_cell - 1
               
            elif (x_loc > RB):      #move particle into cell at right
                p_dist_travled = (RB - p_pos_x)/p_dir_x + kicker
                cell_next = p_mesh_cell + 1
                
            else:                   #move particle in cell
                p_dist_travled = dist
                p_end_trans = 1
                cell_next = p_mesh_cell
                
            p_pos_x = p_pos_x+p_dir_x*p_dist_travled
            p_pos_y = p_pos_y+p_dir_y*p_dist_travled
            p_pos_z = p_pos_z+p_dir_z*p_dist_travled
            
            p_mesh_cell = cell_next
            p_time  += p_dist_travled/p_speed
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_time, p_dist_travled, p_end_trans)



@nb.jit(nopython=True) 
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




def test_Advance():
    L = 1
    dx = .25
    N_m = 4
    
    num_part = 6
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1])
    p_pos_y = 2.1*np.ones(num_part)
    p_pos_z = 3.4*np.ones(num_part)
    
    p_mesh_cell = np.array([-1, 0, 0, 1, 3, 4])
    
    p_dir_x = np.ones(num_part)
    p_dir_x[0] = -1
    p_dir_y = np.zeros(num_part)
    p_dir_z = np.zeros(num_part)
    
    p_speed = np.ones(num_part)
    p_time = np.zeros(num_part)
    p_alive = np.ones(num_part, bool)
    p_alive[5] = False
    
    
    particle_speed = 1
    mesh_total_xsec = np.array([0.1,1,.1,100])
    
    mesh_dist_traveled_squared = np.zeros(N_m)
    mesh_dist_traveled = np.zeros(N_m)
    
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L)
    
    
    assert (np.sum(mesh_dist_traveled) > 0)
    assert (np.sum(mesh_dist_traveled_squared) > 0)
    assert (p_pos_x[0]  == -.01)
    assert (p_pos_x[5]  == 1.1)
    assert (p_pos_x[1:4].all()  > .75)
    
    
        
def test_StillIn():    
    
    num_part = 7
    surface_distances = np.array([0,.25,.75,1])
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1, 1])
    p_alive = np.ones(num_part, bool)
    
    [p_alive, tally_left, tally_right] = StillIn(p_pos_x, surface_distances, p_alive, num_part)
    
    assert(p_alive[0] == False)
    assert(p_alive[5] == False)
    assert(tally_left == 2)
    assert(tally_right == 2)
    assert(p_alive[2:4].all() == True)


if __name__ == '__main__':
    #test_Advance()
    test_StillIn()
    
