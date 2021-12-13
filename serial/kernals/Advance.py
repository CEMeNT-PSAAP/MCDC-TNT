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
            num_part, mesh_total_xsec, mesh_dist_traveled, L):
    kicker = 0.001
    for i in range(num_part):
        #print("particle {0} is now in transport".format(i))
        
        
        flag = 1
        while (flag == 1):
            if (p_pos_x[i] < 0): #exited rhs
                flag = 0            
            elif (p_pos_x[i] > L): #exited lhs
                flag = 0
                
            else:
                dist = -math.log(np.random.random()) / mesh_total_xsec[p_mesh_cell[i]]
                
                x_loc = (p_dir_x[i] * dist) + p_pos_x[i]
                RB = p_mesh_cell[i]+1 * dx
                LB = p_mesh_cell[i]   * dx
                
                if (x_loc < LB):        #move partilce into cell at left
                    dist_traveled = (LB - p_pos_x[i])/p_dir_x[i] + kicker
                   
                elif (x_loc > RB):      #move particle into cell at right
                    dist_traveled = (RB - p_pos_x[i])/p_dir_x[i] + kicker
                    
                else:                   #move particle in cell
                    dist_traveled = dist
                    flag = 0
                    
                p_pos_x[i] += p_dir_x[i]*dist_traveled
                p_pos_y[i] += p_dir_y[i]*dist_traveled
                p_pos_z[i] += p_dir_z[i]*dist_traveled
                mesh_dist_traveled[p_mesh_cell[i]] += dist_traveled
                p_mesh_cell[i] = int(p_pos_x[i]/dx)
                
                #advance particle clock
                p_time[i]  += dist_traveled/p_speed[i]
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled)



def StillIn(p_pos_x, surface_distances, p_alive, num_part):
    tally_left = 0
    tally_right = 0
    for i in range(num_part):
        #exit at left
        if p_pos_x[i] < surface_distances[0]:
            tally_left += 1
            p_alive[i] = False
            
        elif p_pos_x[i] > surface_distances[len(surface_distances)-1]:
            tally_right += 1
            p_alive[i] = False
            
    return(p_alive, tally_left, tally_right)
