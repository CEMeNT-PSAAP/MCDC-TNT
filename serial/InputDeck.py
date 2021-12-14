#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: InputDeck
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import numpy as np


def SimulationSetup():
    #===============================================================================
    # Simulation settings (input deck)
    #===============================================================================
    
    seed = 777
    num_part = 10000 #number of particles to start
    particle_speed = 1 #particle speed
    
    # generations = 1
    nu_new_neutrons = 2 #neutrons/fission
    isotropic = True #isotropic
    
    #===============================================================================
    # Test case 1: Single Reigon
    #===============================================================================
    
    # Lenght_slab = 1
    # surface_distances = np.array([0,Lenght_slab], dtype=np.float32)
    
    # mesh_cell_length = 0.1 #dx
    # N_mesh = int(Lenght_slab/mesh_cell_length)
    
    # cap_xsec = 1/3 #capture crossection
    # scat_xsec = 1/3 #scattering crossection
    # fis_xsec = 1/3 #fission crossection
    
    # #abs_xsec = cap_xsec+fis_xsec #absorption crossection
    # total_xsec = 1 #total crossection
    
    
    # #establishing mesh
    # mesh_scat_xsec = np.zeros(N_mesh, dtype=np.float32)
    # mesh_cap_xsec = np.zeros(N_mesh, dtype=np.float32)
    # mesh_fis_xsec = np.zeros(N_mesh, dtype=np.float32)
    # mesh_total_xsec = np.zeros(N_mesh, dtype=np.float32)

    # for cell in range(N_mesh):
    #     mesh_scat_xsec[cell] = scat_xsec
    #     mesh_cap_xsec[cell] = cap_xsec
    #     mesh_fis_xsec[cell] = fis_xsec
    #     mesh_total_xsec[cell] = total_xsec
    
    
    
    #===============================================================================
    # Test case 2: Three Region
    #===============================================================================
    
    # index refers to region
    cap_xsec = np.array([1/3, 1/3, 1/3]) #capture crossection
    scat_xsec = np.array([2/3, 1/3, 2/3]) #scattering crossection
    fis_xsec = np.array([0,1/3,0]) #fission crossection
    
    #abs_xsec = cap_xsec+fis_xsec #absorption crossection
    total_xsec = np.array([1,1,1]) #total crossection
    
    Lenght_slab = 3
    surface_distances = np.array([0,1,2,Lenght_slab], dtype=np.float32)
    mesh_cell_length = 0.01 #dx
    N_mesh = int(Lenght_slab/mesh_cell_length)
    
    #establishing mesh
    mesh_scat_xsec = np.zeros(N_mesh, dtype=np.float32)
    mesh_cap_xsec = np.zeros(N_mesh, dtype=np.float32)
    mesh_fis_xsec = np.zeros(N_mesh, dtype=np.float32)
    mesh_total_xsec = np.zeros(N_mesh, dtype=np.float32)
    
    for i in range(len(surface_distances)-1):
        for cell in range(int(surface_distances[i]/mesh_cell_length), int(surface_distances[i+1]/mesh_cell_length)):
            mesh_scat_xsec[cell] = scat_xsec[i]
            mesh_cap_xsec[cell] = cap_xsec[i]
            mesh_fis_xsec[cell] = fis_xsec[i]
            mesh_total_xsec[cell] = total_xsec[i]
        
    
    # L = [0,2,4,5] #coordinants of boundaries
    # generation_region = 1
    # regions = 3
    
    return(seed, num_part, particle_speed, nu_new_neutrons, isotropic, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, Lenght_slab, N_mesh, mesh_cell_length, surface_distances)