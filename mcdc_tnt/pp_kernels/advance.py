"""
Name: Advance
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np
import numba as nb

@nb.njit
def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, dt, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_time_cell,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L, max_time):
    """
    Guts of transport is the function that actaully moves particles around, go figure.
    Implements surface tracking with flux (w/ error) via track length estimator

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
    dx : double
        mesh cell width.
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
    num_part : int
        number of particles currently under transport.
    mesh_total_xsec : vector double
        total cross section of every mesh cell (length num_cells).
    mesh_dist_traveled : vector double
        track length estimator tally for use in comp of flux.
    mesh_dist_traveled_squared : TYPE
        distance a particle travels in each cell for use in error with flux.
    L : double
        length of slab.

    Returns
    -------
    Updated PSV with mesh distances.

    """
    kicker = 1e-10
    for i in range(num_part):
        #print(i)
        flag = 1
        cycle_count = 0
        rands = np.random.random(num_part).astype(np.float32)
        while (flag == 1):
            if (p_pos_x[i] < 0): #gone rhs
                flag = 0
            elif (p_pos_x[i] >= L): #gone lhs
                flag = 0
            elif (p_time[i] >= max_time):
                flag = 0
                
            else:
                dist_sampled = -math.log(rands[i]) / mesh_total_xsec[p_mesh_cell[i]]
                
                LB = p_mesh_cell[i] * dx
                RB = LB + dx
                
                TB = ((p_time_cell[i]+1) * dt) - p_time[i]
                #if TB < 0:
                #    print('TIME WAS NEGAITIVE')
                    
                dist_TB = TB * p_speed[i] + kicker
                
                space_cell_inc: int = 0
                if (p_dir_x[i] < 0):
                    dist_B = ((LB - p_pos_x[i])/p_dir_x[i]) + kicker
                    space_cell_inc = -1
                else:
                    dist_B = ((RB - p_pos_x[i])/p_dir_x[i]) + kicker
                    space_cell_inc = 1
                
                
                dist_traveled = min(dist_TB, dist_B, dist_sampled)
                 
                #if dist_traveled < 0:
                #    print('WARNING DISTANCE WAS LESS THAN ZERO')
                increment_time_cell: int = 0
                
                if   dist_traveled == dist_B:      #move partilce into cell at left
                    cell_next = p_mesh_cell[i] + space_cell_inc
                
                elif dist_traveled == dist_sampled: #move particle in cell in time step
                    flag = 0
                    cell_next = p_mesh_cell[i]
                
                elif dist_traveled == dist_TB:
                    p_time_cell[i] += 1
                    cell_next = p_mesh_cell[i]
                    increment_time_cell = 1
                    
                p_pos_x[i] += p_dir_x[i]*dist_traveled
                p_pos_y[i] += p_dir_y[i]*dist_traveled
                p_pos_z[i] += p_dir_z[i]*dist_traveled
                
                #mesh_dist_traveled[p_mesh_cell[i]] += dist_traveled
                #mesh_dist_traveled_squared[p_mesh_cell[i]] += dist_traveled**2
                p_time_cell[i] = int(p_time[i]/dt)
                mesh_dist_traveled[p_mesh_cell[i], p_time_cell[i]] += dist_traveled
                mesh_dist_traveled_squared[p_mesh_cell[i], p_time_cell[i]] += dist_traveled**2
                
                p_mesh_cell[i] = cell_next
                
                #advance particle clock
                p_time[i]  += dist_traveled/p_speed[i] + kicker
                
                p_time_cell[i] += int(p_time[i]/dt)
                
                calc_cell = int(p_time[i]/dt)
                '''
                if p_time_cell[i] != calc_cell:
                    print('Cells did not match for particle {0}'.format(i))
                    print('Calced cell:      {0}'.format(calc_cell))
                    print('Actual listed:    {0}'.format(p_time_cell[i]))
                    print('Particle clock:   {0}'.format(p_time[i]))
                    print('Distance travled: {0}'.format(dist_traveled))
                    print('dist_TB:          {0}'.format(dist_TB))
                    print('dist_B:           {0}'.format(dist_B))
                    print('dist_sampled:     {0}'.format(dist_sampled))
                    print('cycle_count:      {0}'.format(cycle_count))
                    if (increment_time_cell == 1):
                        print('Should have been the time bound')
                    print()
                '''
                cycle_count += 1
                
        #print("Advance Complete:......{0}%".format(int(100*i/num_part)), end = "\r")
    #print()
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_time_cell, mesh_dist_traveled, mesh_dist_traveled_squared)


def StillInSpace(p_pos_x, surface_distances, p_alive, num_part):
    tally_left: int = 0
    tally_right: int = 0
    L = surface_distances[-1]
    for i in range(num_part):
        #exit at left
        if p_pos_x[i] <= 0:
            tally_left += 1
            p_alive[i] = 0
        
        #reflected right
        elif p_pos_x[i] >= L:
            tally_right += 1
            p_alive[i] = 0
            #p_alive[i] = False
            
    return(p_alive, tally_left, tally_right)
    
    
def StillInTime(p_time, max_time, p_alive, num_part):
    
    tally_time: int = 0
    
    for i in range(num_part):
        if p_time[i] >= max_time:
            p_alive[i] = 0
            tally_time +=1
            
    return(p_alive, tally_time)
    
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
    surface_distances = [0,.25,.75,1]
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1, 1])
    p_alive = np.ones(num_part, bool)
    
    [p_alive, tally_left, tally_right] = StillIn(p_pos_x, surface_distances, p_alive, num_part)
    
    assert(p_alive[0] == False)
    assert(p_alive[5] == False)
    assert(tally_left == 2)
    assert(tally_right == 2)
    assert(p_alive[2:4].all() == True)


if __name__ == '__main__':
    test_Advance()
    test_StillIn()
   

    
