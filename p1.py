# -*- coding: utf-8 -*-
### Problem 1

## notes to self:  
    #I need to include a way to give particles different radiuses???
    #Fix the old initialization attempt?
    #I should do a correlation test to find out how many collison I should run between saving data

   


## Import external libraries
import numpy as np 
import matplotlib.pyplot as plt 
import heapq as heap
from scipy import constants as const
from numba import jit 

## Importing my own functions
import plotting 
import initialise_random_particles
import functions
import sim
import test 



 ## Defining parameters
n =5000 #number of particles 
r0 = 0.001 #radius of disk(particles)
r = np.full(n,r0) # array of radiuses 
v0 = 1.0 #velocity of particles
ksi = 1.0 #elastisity coefficient 
m0 = 1.0 #mass of particles 
m = np.full(n, m0) #mass array 
pref = 2/(v0**2) # = kT in Maxwell-Boltzmann sistribution 

#Running the correaltion test 
#test.correlation(v0,m,r,n,ksi)

## Initialize an array of particles and velocities 
x,y,vx,vy = initialise_random_particles.initialise(n,r,v0)


#plotting.particle_pos(x,y,r,'Before sim')
#print(functions.avrg_kin_energy(vx,vy,m))

## Make a histogram of speed (should be a delta func)  
plotting.speed_histogram(vx, vy)    

## Simulating collisions
VX, VY = sim.run(x,y,vx,vy,r,m,100*n,ksi,1)
#np.save('p1_vx_vy.npy',VX,VY)

## New histogram of speed distibution 
#plotting.speed_histogram(vx, vy)   

## Compare to 2D Maxwell-Boltmann distibution 
#plotting.compare_to_Boltzmann(vx,vy,pref)

plotting.compare_to_Boltzmann(VX,VY,pref)


