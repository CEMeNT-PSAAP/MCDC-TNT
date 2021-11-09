#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Testbed_1
breif: Event Based Transient MC for metaprograming exploration
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CeMENT
Date: Nov 8th 2021

current implemented physics:
        -slab geometry
        -monoenergtic
        -isotropic or uniform source direction
        -purely absorbing media
        
"""

import numpy as np
import math


#===============================================================================
# Simulation settings
#===============================================================================

seed = 777
num_part = 100000 #number of particles
particle_speed = 1 #particle speed
abs_xsec = 1.0 #absorption crossection
isotropic = 1 #is it isotropic 1=yes 0=no
L = 2.0


#===============================================================================
# Allocate particle phase space
#===============================================================================

# Position
p_pos_x = np.zeros(num_part, dtype=np.float32)
p_pos_y = np.zeros(num_part, dtype=np.float32)
p_pos_z = np.zeros(num_part, dtype=np.float32)

# Direction
p_dir_x = np.zeros(num_part, dtype=np.float32)
p_dir_y = np.zeros(num_part, dtype=np.float32)
p_dir_z = np.zeros(num_part, dtype=np.float32)

# Speed
p_speed = np.zeros(num_part, dtype=np.float32)

# Time
p_time = np.zeros(num_part, dtype=np.float32)

# Flags
p_alive = np.full(num_part, False)
p_event = np.zeros(num_part, dtype=np.uint8)


#===============================================================================
# Initial setups
#===============================================================================

# Initialize RNG
np.random.seed(seed)


#===============================================================================
# EVENT 0 : Sample particle sources
#===============================================================================

for i in range(num_part):
    # Position
    p_pos_x[i] = 0.0
    p_pos_y[i] = 0.0
    p_pos_z[i] = 0.0

    # Direction
    if isotropic:
        # Sample polar and azimuthal angles uniformly
        mu  = 2.0*np.random.random() - 1.0
        azi = 2.0*np.pi*np.random.random()
	
        # Convert to Cartesian coordinate
        c = (1.0 - mu**2)**0.5
        p_dir_y[i] = np.cos(azi)*c
        p_dir_z[i] = np.sin(azi)*c
        p_dir_x[i] = mu
    else:
        p_dir_x[i] = 1.0
        p_dir_y[i] = 0.0
        p_dir_z[i] = 0.0

    # Speed
    p_speed[i] = particle_speed

    # Time
    p_time[i] = 0.0

    # Flags
    p_alive[i] = True
    p_event[i] = 1


#===============================================================================
# EVENT 1 : Advance
#===============================================================================

for i in range(num_part):
    # Sample distance to collision
    dist = -math.log(np.random.random())/abs_xsec

    # Move particle
    p_pos_x[i] += p_dir_x[i]*dist
    p_pos_y[i] += p_dir_y[i]*dist
    p_pos_z[i] += p_dir_z[i]*dist
    p_time[i]  += dist/p_speed[i]
   

#===============================================================================
# EVENT 10 : Tally score
#===============================================================================

# Count if transmitted
trans = 0
for i in range(num_part):
    if p_pos_x[i]>L:
        trans+=1

print(trans/num_part)
