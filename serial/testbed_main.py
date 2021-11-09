#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Testbed_1
breif: Event Based Transient MC for metaprograming exploration
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 9th 2021
current implemented physics:
        -slab geometry
        -monoenergtic
        -single region
        -isotropic or uniform source direction
        -fission/capture/scatter region
        
"""

import numpy as np
import math

#===============================================================================
# Simulation settings
#===============================================================================

seed = 777
num_part = 1000 #number of particles to start
particle_speed = 1 #particle speed

cap_xsec = 1/3 #capture crossection
scat_xsec = 1/3 #scattering crossection
fis_xsec = 1/3 #fission crossection
abs_xsec = cap_xsec+fis_xsec #absorption crossection
total_xsec = abs_xsec+scat_xsec #total crossection

generations = 10
nu_new_neutrons = 2 #neutrons/fission
isotropic = 1 #is it isotropic 1=yes 0=no

L=2.0

#===============================================================================
# Allocate particle phase space
#===============================================================================
phase_parts = num_part

# Position
p_pos_x = np.zeros(phase_parts, dtype=np.float32)
p_pos_y = np.zeros(phase_parts, dtype=np.float32)
p_pos_z = np.zeros(phase_parts, dtype=np.float32)

# Direction
p_dir_x = np.zeros(phase_parts, dtype=np.float32)
p_dir_y = np.zeros(phase_parts, dtype=np.float32)
p_dir_z = np.zeros(phase_parts, dtype=np.float32)

# Speed
p_speed = np.zeros(phase_parts, dtype=np.float32)

# Time
p_time = np.zeros(phase_parts, dtype=np.float32)

# Flags
p_alive = np.full(phase_parts, False)
p_event = np.zeros(phase_parts, dtype=np.uint8)
"""
p_event: vector of ints to flag what event happened last to this particle
    -1  scattered
    -2  captured
    -3  fission in some previous generation (no fission particles generated)
    -4  fission in the current generation (finision particles made)
    -5  exited the problem on the right
    -6  exited the problem on the left
"""

#===============================================================================
# Initial setups
#===============================================================================

# Initialize RNG
np.random.seed(seed)

norm_scat = scat_xsec/total_xsec
norm_cap = cap_xsec/total_xsec
norm_fis = fis_xsec/total_xsec

#===============================================================================
# EVENT 0 : Sample particle sources
#===============================================================================

for i in range(num_part):
    # Position
    p_pos_x[i] = L*np.random.random()
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


#===============================================================================
# Generation Loop
#===============================================================================
trans = 0
alive_last = num_part
for g in range(generations):
    print("")
    print("===============================================================================")
    print("                             Generation {0}".format(g))
    print("===============================================================================")
    
    #===============================================================================
    # EVENT 1 : Advance
    #===============================================================================
    killed = 0
    
    for i in range(num_part):
        if p_alive[i] == True:
            # Sample distance to collision
            dist = -math.log(np.random.random())/total_xsec
        
            # Move particle
            p_pos_x[i] += p_dir_x[i]*dist
            p_pos_y[i] += p_dir_y[i]*dist
            p_pos_z[i] += p_dir_z[i]*dist
            
            #advance particle clock
            p_time[i]  += dist/p_speed[i]

    #===============================================================================
    # EVENT 2 : Still in problem?
    #===============================================================================
    
    # Count if transmitted
    for i in range(num_part):
        if p_pos_x[i]<0:
            if p_alive[i] == True:
                p_alive[i]=False
                p_event[i]=6
                #killed += 1
            
        if p_pos_x[i]>L:
            if p_alive[i] == True:
                p_event[i] = 5
                p_alive[i] = False
                #killed += 1


    #===============================================================================
    # EVENT 3 : Sample event
    #===============================================================================
    fissions_to_add = 0
    
    for i in range(num_part):
        if p_alive[i] == True:
            event_rand = np.random.random()
            
            #scatter?
            if event_rand < norm_scat:
                p_event[i] = 1
            
            #capture?
            elif norm_scat < event_rand < norm_scat+norm_cap:
                p_event[i] = 2
                p_alive[i] = False
                killed += 1
                
            #fission?
            elif norm_scat+norm_cap < event_rand < norm_scat+norm_cap+norm_fis:
                p_event[i] = 4 #four means fission in this generation will require new particles
                p_alive[i] = False
                killed += 1
                fissions_to_add += nu_new_neutrons
            
            else:
                print("error: event for particle {part} not sorted properly".format(part =i))
                print("random number used: {rand}".foramt(event_rand))
                exit()


    #===============================================================================
    # EVENT 4: Generate fission particles
    #===============================================================================
    
    #allocating new fission particle phase space
    #doing as a two step process in order to minimize the number of append calls
    
    # Position
    p_pos_x_fis = np.zeros(fissions_to_add, dtype=np.float32)
    p_pos_y_fis = np.zeros(fissions_to_add, dtype=np.float32)
    p_pos_z_fis = np.zeros(fissions_to_add, dtype=np.float32)
    
    # Direction
    p_dir_x_fis = np.zeros(fissions_to_add, dtype=np.float32)
    p_dir_y_fis = np.zeros(fissions_to_add, dtype=np.float32)
    p_dir_z_fis = np.zeros(fissions_to_add, dtype=np.float32)
    
    # Speed
    p_speed_fis = np.zeros(fissions_to_add, dtype=np.float32)
    
    # Time
    p_time_fis = np.zeros(fissions_to_add, dtype=np.float32)
    
    # Flags
    p_alive_fis = np.full(fissions_to_add, False)
    p_event_fis = np.zeros(fissions_to_add, dtype=np.uint8)

    k=0 #index for fission temp vectors
    for i in range(num_part):
        if p_event[i] == 4:
            p_event[i] = 3 #fission in some last generation no particles to be generated in the future
            for j in range(nu_new_neutrons):
                # Position
                p_pos_x_fis[k] = p_pos_x[i]
                p_pos_y_fis[k] = p_pos_y[i]
                p_pos_z_fis[k] = p_pos_x[i]
    
                # Direction
                # Sample polar and azimuthal angles uniformly
                mu  = 2.0*np.random.random() - 1.0
                azi = 2.0*np.pi*np.random.random()
                # Convert to Cartesian coordinate
                c = (1.0 - mu**2)**0.5
                p_dir_y_fis[k] = np.cos(azi)*c
                p_dir_z_fis[k] = np.sin(azi)*c
                p_dir_x_fis[k] = mu
                      
                # Speed
                p_speed_fis[k] = particle_speed
                
                # Time
                p_time_fis[k] = p_time[i]
    
                # Flags
                p_alive_fis[k] = True
                
                k+=1
                
    #Append full arrays with fission particles
    p_pos_x = np.append(p_pos_x, p_pos_x_fis)
    p_pos_y = np.append(p_pos_y, p_pos_y_fis)
    p_pos_z = np.append(p_pos_z, p_pos_z_fis)
    
    p_dir_x = np.append(p_dir_x, p_dir_x_fis)
    p_dir_y = np.append(p_dir_y, p_dir_y_fis)
    p_dir_z = np.append(p_dir_z, p_dir_z_fis)
    
    p_speed = np.append(p_speed, p_speed_fis)
    
    p_time = np.append(p_time, p_time_fis)
    
    p_alive = np.append(p_alive, p_alive_fis)
    p_event = np.append(p_event, p_event_fis)
    
    num_part += fissions_to_add
    
    #===============================================================================
    # Criticality & Output (not really an event)
    #===============================================================================
    
    criticality = fissions_to_add/killed
    print("k = {0} (birth/death)".format(criticality))
            
    alive_now =0
    for i in range(num_part):
        if p_alive[i] == True:
            alive_now +=1
    print("k = {0} (pop now/pop last)".format(alive_now/alive_last))
    
    print("population start: {0}".format(alive_last))
    print("population end: {0}".format(alive_now))
    print("particles produced from fission: {0}".format(fissions_to_add))
    print("particles killed: {0}".format(killed))
    print("total particles now stored: {0}".format(num_part))
    
    alive_last = alive_now
    
#===============================================================================
# EVENT 10 : Tally score
#===============================================================================
trans_rhs = 0
for i in range(num_part):
    if p_event[i] == 5:
        trans_rhs += 1

print("")
print("fraction of particles to transit: {0}".format(trans_rhs/num_part))

print('')
print("********************END SIMULATION********************")
print('')
