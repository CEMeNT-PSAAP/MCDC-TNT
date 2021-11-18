#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Advance
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 18th 2021
"""
import math
import numpy as np

def Advance(p_pos_x, p_pos_y, p_pos_z, p_region, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, total_xsec):
    for i in range(num_part):
        # Sample distance to collision
        dist = -math.log(np.random.random())/total_xsec[p_region[i]]
    
        # Move particle
        p_pos_x[i] += p_dir_x[i]*dist
        p_pos_y[i] += p_dir_y[i]*dist
        p_pos_z[i] += p_dir_z[i]*dist
        
        #advance particle clock
        p_time[i]  += dist/p_speed[i]
    
    return(p_pos_x, p_pos_y, p_pos_z, p_region, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time)