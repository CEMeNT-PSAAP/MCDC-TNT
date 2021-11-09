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

np.random.seed(777)
isotropic = 0 #is it isotropic 1=yes 0=no

num_part = 1000 #number of particles
particle_speed = 1 #particle speed
abs_xsec = 0.05 #absorption crossection
iabs_xsec = 1/abs_xsec

#index of vectors corresponds to individual particles
x = np.zeros(num_part)
y = np.zeros(num_part)
z = np.zeros(num_part)

if isotropic == 0:
    i_hat = np.ones(num_part)
    j_hat = np.zeros(num_part)
    k_hat = np.zeros(num_part)
else:
    i_hat = np.random.random()*np.ones(num_part)
    j_hat = np.random.random()*np.ones(num_part)
    k_hat = np.random.random()*np.ones(num_part)

speed = particle_speed*np.ones(num_part)

alive_flag = np.ones(num_part)
#for a given cycle what is the event (all absorption right now)
event_flag = np.ones(num_part)

clocks = np.zeros(num_part)

L = 2

trans = 0

#sample distance to collision
for i in range(num_part):
    dist = -math.log(np.random.random())*iabs_xsec
    x[i] = x[i] + i_hat[i]*dist
    y[i] = y[i] + j_hat[i]*dist
    z[i] = z[i] + k_hat[i]*dist
    
    #update clock of individual particles
    clocks[i] += dist/speed[i]
    
    #count if transmitted
    if x[i]>L:
        trans+=1

print(trans/num_part)