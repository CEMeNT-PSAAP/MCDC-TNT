#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Testbed_1
breif: Event Based Transient MC for metaprograming exploration
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 9th 2021
current implemented physics:
        -slab geometry
        -multiregion (only one generating)
        -monoenergtic
        -isotropic or uniform source direction
        -fission/capture/scatter region
        -purging the dead
"""

import numpy as np
import math

#===============================================================================
# Simulation settings (input deck)
#===============================================================================

seed = 777
num_part = 100000 #number of particles to start
init_particle = num_part
particle_speed = 1 #particle speed

generations = 1
nu_new_neutrons = 2 #neutrons/fission
isotropic = 1 #is it isotropic 1=yes 0=no

# Test case 1
cap_xsec = np.array([1/3]) #capture crossection
scat_xsec = np.array([1/3]) #scattering crossection
fis_xsec = np.array([1/3]) #fission crossection

#abs_xsec = cap_xsec+fis_xsec #absorption crossection
total_xsec = np.array([1]) #total crossection

L = [0,2] #x coordinants of boundaries
generation_region = 0
regions = 1

# Test case 2
#index refers to region
# cap_xsec = np.array([1/3, 1/3, 1/3]) #capture crossection
# scat_xsec = np.array([2/3, 1/3, 2/3]) #scattering crossection
# fis_xsec = np.array([0,1/3,0]) #fission crossection

# #abs_xsec = cap_xsec+fis_xsec #absorption crossection
# total_xsec = np.array([1,1,1]) #total crossection
#L = [0,2,4,5] #coordinants of boundaries
#generation_region = 1
# regions = 3

#===============================================================================
# Allocate particle phase space
#===============================================================================

phase_parts = 5*num_part

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

# Region
p_region = np.zeros(phase_parts, dtype=np.uint8)

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

L_gen = L[1]-L[0]
x_rhs_gen = L[0]

for i in range(num_part):
    # Position
    p_pos_x[i] = x_rhs_gen + L_gen*np.random.random()
    p_region[i] = generation_region
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
g = 0
alive = num_part
total_particles_to_live = num_part
trans_lhs = 0
trans_rhs = 0
while alive > 0:
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
            dist = -math.log(np.random.random())/total_xsec[p_region[i]]
        
            # Move particle
            p_pos_x[i] += p_dir_x[i]*dist
            p_pos_y[i] += p_dir_y[i]*dist
            p_pos_z[i] += p_dir_z[i]*dist
            
            #advance particle clock
            p_time[i]  += dist/p_speed[i]

    #===============================================================================
    # EVENT 2 : Still in problem/ update region
    #===============================================================================
    
    # Count if transmitted
    for i in range(num_part):
        #leak left
        if p_pos_x[i]<L[0]:
            if p_alive[i] == True:
                p_alive[i]=False
                p_event[i]=6
                #killed += 1
        #leak right
        elif p_pos_x[i]>L[1]:
            if p_alive[i] == True:
                p_event[i] = 5
                p_alive[i] = False
                #killed += 1
        else:
            for l in range(regions):
                if (L[l] < p_pos_x[i] < L[l+1]):
                    p_region[i] = l
                

    #===============================================================================
    # EVENT 3 : Sample event
    #===============================================================================
    fissions_to_add = 0
    
    for i in range(num_part):
        if p_alive[i] == True:
            event_rand = np.random.random()
            
            #scatter?
            if event_rand < norm_scat[p_region[i]]:
                p_event[i] = 1
            
            #capture?
            elif norm_scat[p_region[i]] < event_rand < norm_scat[p_region[i]]+norm_cap[p_region[i]]:
                p_event[i] = 2
                p_alive[i] = False
                killed += 1
                
            #fission?
            elif norm_scat[p_region[i]]+norm_cap[p_region[i]] < event_rand < norm_scat[p_region[i]]+norm_cap[p_region[i]]+norm_fis[p_region[i]]:
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
    

    k=0 #index for fission temp vectors
    for i in range(num_part):
        if p_event[i] == 4:
            p_event[i] = 3 #fission in some last generation no particles to be generated in the future
            for j in range(nu_new_neutrons):
                # Position
                p_pos_x[k+num_part] = p_pos_x[i]
                p_region[k+num_part] = p_region[i]
                p_pos_y[k+num_part] = p_pos_y[i]
                p_pos_z[k+num_part] = p_pos_x[i]
                
                # Direction
                # Sample polar and azimuthal angles uniformly
                mu  = 2.0*np.random.random() - 1.0
                azi = 2.0*np.pi*np.random.random()
                # Convert to Cartesian coordinate
                c = (1.0 - mu**2)**0.5
                p_dir_y[k+num_part] = np.cos(azi)*c
                p_dir_z[k+num_part] = np.sin(azi)*c
                p_dir_x[k+num_part] = mu
                      
                # Speed
                p_speed[k+num_part] = particle_speed
                
                # Time
                p_time[k+num_part] = p_time[i]
    
                # Flags
                p_alive[k+num_part] = True
                
                k+=1
                
    num_part += k
    total_particles_to_live += k
    
    #===============================================================================
    # Criticality & Output (not really an event)
    #===============================================================================
    
    # criticality = fissions_to_add/killed
    # print("k = {0} (birth/death)".format(criticality))
            
    # alive_now =0
    # for i in range(num_part):
    #     if p_alive[i] == True:
    #         alive_now +=1
    # print("k = {0} (pop now/pop last)".format(alive_now/alive_last))
    
    
    #===============================================================================
    # EVENT 10 : Tally score
    #===============================================================================
    for i in range(num_part):
        if p_event[i] == 5:
            trans_rhs += 1
    for i in range(num_part):
        if p_event[i] == 6:
            trans_lhs += 1
    
    
    #===============================================================================
    # Event 5: Purge the dead
    #===============================================================================
    kept = 0
    for i in range(num_part):
        if p_alive[i] == True:
            p_pos_x[k] = p_pos_x[i]
            p_pos_y[k] = p_pos_y[i]
            p_pos_z[k] = p_pos_z[i]
            
            # Direction
            p_dir_x[k] = p_dir_x[i]
            p_dir_y[k] = p_dir_y[i]
            p_dir_z[k] = p_dir_z[i]
            
            # Speed
            p_speed[k] = p_speed[i]
            
            # Time
            p_time[k] = p_time[i]
            
            # Regions
            p_region[k] = p_region[i]
            
            # Flags
            p_alive[k] = p_alive[i] 
            p_event[k] = p_event[i]
            kept +=1
            
    num_part = kept
    alive = num_part
    # print("population start: {0}".format(alive_last))
    # print("population end: {0}".format(alive_now))
    print("particles produced from fission: {0}".format(fissions_to_add))
    print("particles killed: {0}".format(killed))
    print("total particles now stored: {0}".format(num_part))
    # alive_last = alive_now
    
    g+=1


print("")
print("leak left: {0}".format(trans_lhs/init_particle))
print("")
print("leak right: {0}".format(trans_rhs/init_particle))
print("")
print('')
print("********************END SIMULATION********************")
print('')
