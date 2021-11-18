#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: CleanUp
breif: Misc functions for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 18th 2021
"""

def UpdateRegion(p_pos_x, p_region, p_alive, p_event, num_part, regions, L):
    # Count if transmitted
    for i in range(num_part):
        #leak left
        if p_pos_x[i]<L[0]:
            p_alive[i]=False
            p_event[i]=6
            #killed += 1
        #leak right
        elif p_pos_x[i]>L[1]:
            p_event[i] = 5
            p_alive[i] = False
            #killed += 1
        else:
            for l in range(regions):
                if (L[l] < p_pos_x[i] < L[l+1]):
                    p_region[i] = l
                    
    return(p_alive, p_event, p_region)



def BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_region, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, p_event, num_part):
    kept = 0
    for i in range(num_part):
        if p_alive[i] == True:
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
            p_region[kept] = p_region[i]
            
            # Flags
            p_alive[kept] = p_alive[i] 
            p_event[kept] = p_event[i]
            kept +=1
    return(p_pos_x, p_pos_y, p_pos_z, p_region, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, kept)