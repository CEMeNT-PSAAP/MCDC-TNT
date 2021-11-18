#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: InputDeck
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 18th 2021
"""

import numpy as np


def SimulationSetup():
    #===============================================================================
    # Simulation settings (input deck)
    #===============================================================================
    
    seed = 777
    num_part = 100000 #number of particles to start
    particle_speed = 1 #particle speed
    
    # generations = 1
    nu_new_neutrons = 2 #neutrons/fission
    isotropic = True #is it isotropic 1=yes 0=no
    
    # Test case 1
    cap_xsec = np.array([1/3]) #capture crossection
    scat_xsec = np.array([1/3]) #scattering crossection
    fis_xsec = np.array([1/3]) #fission crossection
    
    #abs_xsec = cap_xsec+fis_xsec #absorption crossection
    total_xsec = np.array([1]) #total crossection
    
    L = [0,2] #x coordinants of boundaries
    generation_region = 0
    regions = 1
    
    # # Test case 2
    # # index refers to region
    # cap_xsec = np.array([1/3, 1/3, 1/3]) #capture crossection
    # scat_xsec = np.array([2/3, 1/3, 2/3]) #scattering crossection
    # fis_xsec = np.array([0,1/3,0]) #fission crossection
    
    # #abs_xsec = cap_xsec+fis_xsec #absorption crossection
    # total_xsec = np.array([1,1,1]) #total crossection
    # L = [0,2,4,5] #coordinants of boundaries
    # generation_region = 1
    # regions = 3
    
    return(seed, num_part, particle_speed, nu_new_neutrons, isotropic, cap_xsec, scat_xsec, fis_xsec, total_xsec, L, generation_region, regions)