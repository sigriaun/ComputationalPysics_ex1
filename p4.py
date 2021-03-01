## Import external libraries
import numpy as np 
import matplotlib.pyplot as plt 
import heapq as heap


## Importing my own functions
import plotting 
import initialise_random_particles
import functions
import sim

n = 3000
packing_factor = 0.5
r0 = np.sqrt(0.5*0.5/np.pi/(n-1))

m0 = 2
m_proj = 25
r_proj = r0*5
v0 = 5
ksi = 0.5

m = np.full(n,m0)
m[0] = m_proj

r = np.full(n,r0)
r[0] = r_proj 

#Step one: initialize particles and test the packing factor 
x,y,vx,vy = initialise_random_particles.initialise_p4(n, v0, r)
#np.save('p4data3.npy',[x,y,vx,vy])

#Loading saved particle positions 
[x,y,vx,vy] = np.load('p4data3.npy')

#copying initial position to be avle to compare against after the simulation.
x_ini = np.copy(x)
y_ini = np.copy(y)


plotting.particle_pos_p4(x,y)

sim.run(x,y,vx,vy,r,m,1000000,ksi,4)

plotting.particle_pos_p4(x,y)
crater_size = functions.crater_size(x_ini,y_ini,x,y)
print('crater size',crater_size)


#sweep of velocity
# crater_sizes = []
# m0s = np.linspace(2,50,10)
# for m0 in m0s:  
#     [x,y,vx,vy] = np.load('p4data3.npy')
#     m[0] = m0
#     sim.run4(x,y,vx,vy,r,m,1000000,ksi,4)
#     cs = functions.crater_size(x_ini,y_ini,x,y)
#     plotting.particle_pos_p4(x,y)
#     crater_sizes.append(cs)
#     print('crater size',cs)

# plt.plot(m0s,crater_sizes)
# plt.show()

