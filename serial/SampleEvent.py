#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: SampleEvent
breif: Samples events for particles provided in a phase space for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""
import numpy as np


def SampleEvent(p_mesh_cell, p_event, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index, capture_event_index, fission_event_index, num_part, nu_new_neutrons):
    fissions_to_add = 0
    scat_count = 0
    cap_count = 0
    fis_count = 0
    killed = 0
    
    for i in range(num_part):
        if p_alive[i] == True:
            event_rand = np.random.random()
            
            #scatter?
            if event_rand < mesh_scat_xsec[p_mesh_cell[i]]:
                p_event[i] = 1
                scatter_event_index[scat_count] = i
                scat_count += 1
                
            
            #capture?
            elif mesh_scat_xsec[p_mesh_cell[i]] < event_rand < mesh_scat_xsec[p_mesh_cell[i]]+mesh_cap_xsec[p_mesh_cell[i]]:
                p_event[i] = 2
                p_alive[i] = False
                killed += 1
                capture_event_index[cap_count] = i
                cap_count +=1
                
            #fission?
            elif mesh_scat_xsec[p_mesh_cell[i]] + mesh_cap_xsec[p_mesh_cell[i]] < event_rand < mesh_scat_xsec[p_mesh_cell[i]] + mesh_cap_xsec[p_mesh_cell[i]] + mesh_fis_xsec[p_mesh_cell[i]]:
                p_event[i] = 4 #four means fission in this generation will require new particles
                p_alive[i] = False
                killed += 1
                fissions_to_add += nu_new_neutrons
                fission_event_index[fis_count] = i
                fis_count +=1
                
            else:
                print("error: event for particle {part} not sorted properly".format(part =i))
                print("random number used: {rand}".foramt(event_rand))
                exit()
                
    return(scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count)
