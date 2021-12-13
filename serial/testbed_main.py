#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Testbed_1
breif: Event Based Transient MC for metaprograming exploration
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
current implemented physics:
        -slab geometry
        -multiregion
        -surface tracking
        -monoenergtic
        -isotropic or uniform source direction
        -fission/capture/scatter region
        -purging the dead
"""
# import sys
# sys.path.append('/home/jack/Documents/testbed/serial/kernals/')
import numpy as np
from InputDeck import SimulationSetup
from kernals.SourceParticles import *
from kernals.Advance import *
from kernals.SampleEvent import *
from kernals.FissionsAdd import *
from kernals.CleanUp import *
from kernals.Scatter import *


#===============================================================================
# Simulation Setup
#===============================================================================

[seed, num_part, particle_speed, nu_new_neutrons, isotropic, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, L, N_mesh, dx, surface_distances] = SimulationSetup()

#===============================================================================
# Allocate particle phase space
#===============================================================================

phase_parts = 2*num_part #see note about data storage

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
p_mesh_cell = np.zeros(phase_parts, dtype=int)

# Flags
p_alive = np.full(phase_parts, False)
p_event = np.zeros(phase_parts, dtype=np.uint8)


p_region = np.zeros(phase_parts, dtype=int)
#mesh_particle_index = np.zeros([N_mesh, phase_parts], dtype=np.uint8)


scatter_event_index = np.zeros(phase_parts, dtype=np.uint8)
capture_event_index = np.zeros(phase_parts, dtype=np.uint8)
fission_event_index = np.zeros(phase_parts, dtype=np.uint8)

"""
p_event: vector of ints to flag what event happened last to this particle
    -1  scattered
    -2  captured
    -3  fission in some previous generation (no fission particles generated)
    -4  fission in the current generation (finision particles made)
    -5  exited the problem on the right
    -6  exited the problem on the left
"""


#seed, num_part, particle_speed, nu_new_neutrons, isotropic, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, L, generation_region, regions = SimulationSetup()

#===============================================================================
# Initial setups
#===============================================================================

# Initialize RNG
np.random.seed(seed)

init_particle = num_part
meshwise_fission_pdf = np.zeros(N_mesh, dtype=np.float32)

total_mesh_fission_xsec = sum(mesh_fis_xsec)
for cell in range(N_mesh):
    meshwise_fission_pdf[cell] = mesh_fis_xsec[cell]/total_mesh_fission_xsec
    
    mesh_cap_xsec[cell] = mesh_cap_xsec[cell] / mesh_total_xsec[cell]
    mesh_scat_xsec[cell] = mesh_scat_xsec[cell] / mesh_total_xsec[cell]
    mesh_fis_xsec[cell] = mesh_fis_xsec[cell] / mesh_total_xsec[cell]

meshwise_fission_pdf /= sum(meshwise_fission_pdf)
mesh_dist_traveled = np.zeros(N_mesh, dtype=np.float32)

#===============================================================================
# EVENT 0 : Sample particle sources
#===============================================================================

p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive = SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_part, meshwise_fission_pdf, particle_speed, isotropic=True)

#===============================================================================
# Generation Loop
#===============================================================================
trans = 0
g = 0
alive = num_part
trans_lhs = 0
trans_rhs = 0
while alive > 0:
    print("")
    print("===============================================================================")
    print("                             Event Cycle {0}".format(g))
    print("===============================================================================")
    print("particles alive at start of event cycle {0}".format(num_part))
    
    
    #===============================================================================
    # EVENT 1 : Advance
    #===============================================================================
    killed = 0
    alive_cycle_start = num_part
    
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled] = Advance(
            p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, 1)
    
    #===============================================================================
    # EVENT 2 : Still in problem
    #===============================================================================
                
    [p_alive, tally_left_t, tally_right_t] = StillIn(p_pos_x, surface_distances, p_alive, num_part)
    
    trans_lhs += tally_left_t
    trans_rhs += tally_right_t
    
    
    #===============================================================================
    # EVENT 3 : Sample event
    #===============================================================================
    
    [scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count] = SampleEvent(
            p_mesh_cell, p_event, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index,
            capture_event_index, fission_event_index, num_part, nu_new_neutrons)
    
    fissions_to_add = (fis_count)*nu_new_neutrons
    
    killed += cap_count+fis_count
    #===============================================================================
    # EVENT 3 : Scatter
    #===============================================================================
    
    [p_dir_x, p_dir_y, p_dir_z] = Scatter(scatter_event_index, scat_count, p_dir_x, p_dir_y, p_dir_z)
    
    
    #===============================================================================
    # EVENT 4: Generate fission particles
    #===============================================================================
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, 
     p_time, p_alive, particles_added_fission] = FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                              p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                              p_time, p_alive, fis_count, nu_new_neutrons, 
                                              fission_event_index, num_part, particle_speed)
                                              
    num_part += particles_added_fission
                                              
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
    # Event 5: Purge the dead
    #===============================================================================
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, 
     p_time, p_alive, kept] = BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                               p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                               p_time, p_alive, p_event, num_part)
                                               
    num_part = kept
    alive = num_part
                         
    print("")
    
    # print(max(p_mesh_cell[0:num_part]))
    
    #===============================================================================
    # Step Output
    #===============================================================================
    
    # the sum of all debits from functional operations should be the number of
    #currently alive particles
    account = alive_cycle_start + particles_added_fission - fis_count - cap_count - tally_left_t - tally_right_t
    if account != num_part:
        print("ERROR PARTICLES HAVE BEEN UNACCOUNTED FOR")
    
    print("{0} particles are produced from {1} fission events".format(fissions_to_add, fis_count))
    print("particles captured:  {0}".format(cap_count))
    print("particles scattered: {0}".format(scat_count))
    print("particles leaving left:    {0}".format(tally_left_t))
    print("particles leaving right:   {0}".format(tally_right_t))
    print("total particles now alive and stored: {0}".format(num_part))
    
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
