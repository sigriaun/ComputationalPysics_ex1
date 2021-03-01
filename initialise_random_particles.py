import numpy as np 
import sys 

import plotting


def move_overlapping_part(x,y,r,n,y_maks=1):  
    k = 0 
    for i in range(1,n):
        overlapping = True 
        while overlapping:
            k= k+1
            rij_sq = (x[:i]-x[i])**2 + (y[:i]-y[i])**2      
            Rij_sq = (r[i] + r[:i])**2
            overlaping = rij_sq < Rij_sq ;
            if k%10 == 0: 
                print(i, k)
                
            if True in overlaping: 
                x[i] = r[i] + (1-2*r[i])*np.random.rand()
                y[i] = r[i] + (y_maks-2*r[i])*np.random.rand()  
            else: 
                overlapping = False
            if k > 1000*n: 
                plotting.particle_pos(x,y,r,'initializing')
                sys.exit('could not initialize particle position') 


def initialise(n,r,v0):
    x = r + (1-2*r)*np.random.rand(n)
    y = r + (1-2*r)*np.random.rand(n)

    theta = 2*np.pi*np.random.rand(n)
    
    vx = v0*np.cos(theta) 
    vy = v0*np.sin(theta)    
    
    all_ind = np.arange(n)

    move_overlapping_part(x,y,r,n)
    
    return x,y,vx,vy


def initialise_p4(n,v0,r):
    vx = np.zeros(n)
    vy = np.zeros(n)
    vy[0] = -v0
    
    x = r + (1  -2*r)*np.random.rand(n)
    y = r + (0.5-2*r)*np.random.rand(n)
    x[0] = 0.5
    y[0] = 0.75
    
    
    move_overlapping_part(x,y,r,n,0.5)
    
    return x,y,vx,vy



