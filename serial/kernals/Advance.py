#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Advance
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np

def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec):
    for i in range(num_part):
        # Sample distance to collision
        dist = -math.log(np.random.random()) / mesh_total_xsec[p_mesh_cell[i]]
        dist_x = p_dir_x[i] * dist
        
            
        #move particle into cell at right
        if (dist_x > dx*p_mesh_cell[i+1]):
            p_mesh_cell[i] += 1
            p_pos_x[i] = dx*np.random.random()
            
        #move partilce into cell at left
        elif (dist_x < dx*p_mesh_cell[i]):
            p_mesh_cell[i] -= 1
            p_pos_x[i] = dx*np.random.random()
        
        # Move particle in cell
        else:
            p_pos_x[i] += dist_x
            p_pos_y[i] += p_dir_y[i]*dist
            p_pos_z[i] += p_dir_z[i]*dist
        
        #advance particle clock
        p_time[i]  += dist/p_speed[i]
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time)



def StillIn(p_mesh_cell, p_alive, num_part, N_cell):
    for i in range(num_part):
        tally_left = 0
        tally_right = 0
        
        #exit at left
        if p_mesh_cell[i] < 0:
            tally_left += 1
            p_alive[i] = False
            
        elif p_mesh_cell[i] > (N_cell - 1):
            tally_right += 1
            p_alive[i] = False
            
    return(p_alive, tally_left, tally_right)