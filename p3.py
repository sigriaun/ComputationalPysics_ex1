# -*- coding: utf-8 -*-

## Import external libraries
import numpy as np 
import matplotlib.pyplot as plt 
import heapq as heap
import matplotlib.pyplot as plt 


## Importing my own functions
import plotting 
import initialise_random_particles
import sim
import functions


## Setting paramters
m0 = 1

n = 2500
N = n*2 

r0 = 0.001
r = np.full(N,r0)

v0 = 1

m = np.zeros(N)
m[:n] = m0 
m[n:] = 4* m0



#run for xi = 1 
ksis = [1,0.9,0.8]

for ksi in ksis: 
    x,y,vx,vy = initialise_random_particles.initialise(N,r,v0) 
    p3_data = sim.run(x,y,vx,vy,r,m,15*n,ksi,3)
    plotting.kinetic_energy_aft(p3_data)
