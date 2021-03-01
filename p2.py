# -*- coding: utf-8 -*-

## Import external libraries
import numpy as np 
import matplotlib.pyplot as plt 
import heapq as heap


## Importing my own functions
import plotting 
import initialise_random_particles
import functions
import sim

## Setting paramters
n = 2500 # number of particles per particle type
N = n*2 # total number of particles
ksi = 1
v0 = 1
r0 = 0.001
r = np.full(N,r0)
m0 = 1
m = np.zeros(N)
m[:n] = m0 
m[n:] = 4* m0

## Initializing particle position and velocities
x,y,vx,vy = initialise_random_particles.initialise(N,r,v0)

##Calculating initaly average speed and kinetic energy 
as1_0 = functions.avrg_speed(vx[:n], vy[:n])
as2_0 = functions.avrg_speed(vx[n:], vy[n:])
ak1_0 = functions.avrg_kin_energy(vx[:n], vy[:n], m0)
ak2_0 = functions.avrg_kin_energy(vx[n:], vy[n:], 4*m0)

## Plotting histograms before running simulation
plotting.speed_histogram2(vx[:n], vy[:n],vx[n:], vy[n:])

## Running simulation 

sim.run(x,y,vx,vy,r,m,10*n,ksi,2)
#np.save('p2data.npy',[vx,vy])
## the saved collison data
#[vx,vy] = np.load('p2data.npy')

## Plotting histogram after simuation 
plotting.speed_histogram2(vx[:n], vy[:n],vx[n:], vy[n:])

# print('1',VX[:,:n])
# print('2',VX[:n,:])
# print('3',VX)
#plotting.speed_histogram2(VX[:,:n], VY[:,:n],VX[:,n:], VY[:,n:])

## Calculating average kinetic energy and average speed after simulation
as1 = functions.avrg_speed(vx[:n], vy[:n])
as2 = functions.avrg_speed(vx[n:], vy[n:])
ak1 = functions.avrg_kin_energy(vx[:n], vy[:n], m0)
ak2 = functions.avrg_kin_energy(vx[n:], vy[n:], 4*m0)


print('average speed:  \n 1 start, finish', as1_0,as1, '\n 2 start, finish', as2_0, as2)
print('average kinetic energy \n 1 start, finish ', ak1_0,ak1, '\n 2 start, finish', ak2_0, ak2)




