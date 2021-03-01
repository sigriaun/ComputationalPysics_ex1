# -*- coding: utf-8 -*-
import numpy as np 
import heapq as heap
from numba import jit

import matplotlib.pyplot as plt 
import plotting 
import functions

## Run system, at least n collisions 
@jit(nopython = True)
def find_next_collison(x,y,vx,vy,r,i):#x,y,vx,vy,r,i
    
    
    if vx[i] > 0: 
        deltaTx = (1-r[i]-x[i])/vx[i]
    elif vx[i] <0: 
        deltaTx = (r[i] - x[i])/vx[i] 
    else: 
        deltaTx = np.inf
        
    if vy[i] > 0: 
        deltaTy = (1-r[i]-y[i])/vy[i]
    elif vy[i]<0:
        deltaTy = (r[i] - y[i])/vy[i]
    else: 
        deltaTy = np.inf
        
    deltaXs = np.zeros(  ( 2 ,  len(x)) )
    deltaVs = np.zeros(  ( 2 ,  len(x)) )
    deltaXs = [  x[:] -  x[i] ,  y[:] -  y[i] ]
    deltaVs = [ vx[:] - vx[i] , vy[:] - vy[i] ]
    
    R_sq = (r[:] + r[i])**2
 
    xdotv = deltaXs[0][:]*deltaVs[0][:] + deltaXs[1][:]*deltaVs[1][:]
    xdotx = deltaXs[0][:]*deltaXs[0][:] + deltaXs[1][:]*deltaXs[1][:]
    vdotv = deltaVs[0][:]*deltaVs[0][:] + deltaVs[1][:]*deltaVs[1][:]
    
    d = np.zeros(len(x))
    d[:] = xdotv[:]**2 - vdotv[:] * ( xdotx[:] - R_sq[:] )
    
    ind = np.full( len(x) , False)
    ind1 =  xdotv < 0 
    ind2  =   d > 0  
    #ind = np.all([ind1,ind2],axis=0)
    #print(ind)
    ind = ind1*ind2
    ind_test = ind1*ind2
    #print(ind_test)
    ind[i] = False
    
    false_ind = np.full(len(x),True)
    ind3 =  xdotv >= 0 
    ind4  =   d <= 0  
    false_ind = ind3 + ind4
   
    deltaTs = np.zeros(len(x))
    deltaTs[ind] = - (xdotv[ind] + np.sqrt(d[ind]) )/vdotv[ind]
    deltaTs[ind3] = np.inf
    deltaTs[ind4] = np.inf
    #deltaTs[false_ind] = np.inf
    
    # 
    # deltaTpm = np.max(deltaTs) 
    # maxT = np.max([deltaTpm, deltaTy, deltaTx])
    # deltaTs[false_ind] = maxT +1
    # if deltaTx == -1: 
    #     deltaTx = maxT +1 
    # if deltaTy == -1: 
    #     deltaTy = maxT +1
    
    #print(deltaTs)
    part_ind = np.argmin(deltaTs)
    deltaTp = deltaTs[part_ind]
   
    if deltaTx < deltaTp and deltaTx < deltaTy: 
        return deltaTx,-1    
    elif deltaTy < deltaTp:
        return deltaTy,-2
    else: 
        return deltaTp,part_ind
    
#@jit(nopython = True) does not work 
def initial_collisons(x,y,vx,vy,r,colliding_with):
    collisons = [] 
    for i in range(len(x)): 
        dt, j = find_next_collison(x,y,vx,vy,r,i)
        if dt == np.inf:
            colliding_with[i] = -3
        
        else:
            colliding_with[i] = j 
            heap.heappush( collisons,[dt,i,j,0,0] )      
    return collisons 

@jit(nopython=True )
def move_all_parts_deltaT(x,y,vx,vy,col_time):
    x[:] = x[:] + vx[:]*col_time
    y[:] = y[:] + vy[:]*col_time
    x = x + vx*col_time
    y = y + vy*col_time


@jit(nopython = True)
def collide_part_wall(x,y,vx,vy,r,i,j,dt,ksi):
    move_all_parts_deltaT(x,y,vx,vy,dt) 
    
    if j == -1:
        vx[i] = -ksi*vx[i]
        vy[i] = ksi*vy[i] 
    elif j == -2:
        vx[i] = ksi*vx[i]
        vy[i] = -ksi*vy[i]        
        

@jit(nopython = True)
def collide_part_part(x,y,vx,vy,r,m,i,j,dt,ksi): 
    move_all_parts_deltaT(x,y,vx,vy,dt)

    #updating velocities
    Rij_sq = (r[i]+r[j])**2

    vxi = vx[i] + ( (1+ksi)* m[j]/(m[j]+m[i]) *( (x[j]-x[i])*(vx[j]-vx[i])  + (y[j]-y[i])*(vy[j]-vy[i]) )/Rij_sq ) * (x[j]-x[i])
    vyi = vy[i] + ( (1+ksi)* m[j]/(m[j]+m[i]) *( (x[j]-x[i])*(vx[j]-vx[i])  + (y[j]-y[i])*(vy[j]-vy[i]) )/Rij_sq ) * (y[j]-y[i])
    # print('second term of vx[i]',( (1+ksi)* m[j]/(m[j]+m[i]) *( (x[j]-x[i])*(vx[j]-vx[i])  + (y[j]-y[i])*(vy[j]-vy[i]) )/Rij_sq ) * (x[j]-x[i]))
    # print('second term of vx[j]',( (1+ksi)* m[i]/(m[i]+m[j]) *( (x[j]-x[i])*(vx[j]-vx[i])  + (y[j]-y[i])*(vy[j]-vy[i]) )/Rij_sq ) * (x[j]-x[i]))
    # print('deltax',x[j]-x[i],y[j]-y[i])
    vxj = vx[j] - ( (1+ksi)* m[i]/(m[i]+m[j]) *( (x[j]-x[i])*(vx[j]-vx[i])  + (y[j]-y[i])*(vy[j]-vy[i]) )/Rij_sq ) * (x[j]-x[i])
    vyj = vy[j] - ( (1+ksi)* m[i]/(m[i]+m[j]) *( (x[j]-x[i])*(vx[j]-vx[i])  + (y[j]-y[i])*(vy[j]-vy[i]) )/Rij_sq ) * (y[j]-y[i])
    #print('vdotx',vdotx)
    
    vx[i] = vxi
    vy[i] = vyi
    vx[j] = vxj
    vy[j] = vyj
    
def add_new_coll(x,y,vx,vy,r,time,collisons,i,colliding_with,ncp):
    #comment: I know thatl and k could be -1 or -2, 
    #but in that case I do not care about ncp[k] anyways so I say it's fine
    ind = np.argwhere(colliding_with == i)

    
    dt, l = find_next_collison(x,y,vx,vy,r,i)
    colli = [dt+time,i,l,ncp[i],ncp[l]]
    heap.heappush(collisons,colli)
    colliding_with[i] = l
       
    for j_arr in ind: #go through all particles that where to collide with particle i 
        j = j_arr[0]
        dt, k  = find_next_collison(x,y,vx,vy,r,j)
        collj = [dt+time,j,k,ncp[j],ncp[k]] 
        heap.heappush(collisons,collj)
        colliding_with[j] = k
    

def run(x,y,vx,vy,r,m,nc,ksi,prob): 
    print('sim... initializing collisions')
    
    colliding_with = np.zeros(len(x)) #trying this! saving which particele the collison is between 
    
    collisons = initial_collisons(x,y,vx,vy,r,colliding_with)

    #print(colliding_with)
    
    ncp = np.zeros(len(x))  #number of collisions per particle 
    
    n = len(x)
    cc = 0 # collison count 
    time = 0 
    ppcc = 0 #particle-particle collision count
    pwcc = 0 #particle-wall collision count 
    
   
    if prob == 4:
        kin_energy_start = functions.tot_kin_energy(vx, vy, m)
    if prob == 3:
        time_list = [0] # list to fill in time 
        # list to fill in kinetic energy 
        nhalf = n//2
        K_all = [functions.avrg_kin_energy(vx,vy,m)] 
        K1 = [functions.avrg_kin_energy(vx[:nhalf],vy[:nhalf],m[:nhalf])]
        K2 = [functions.avrg_kin_energy(vx[nhalf:],vy[nhalf:],m[nhalf:])]
        
    if prob == 1: 
        nf = 5 # n*nf = first time I want to take samples
        VX = np.zeros((nc//n-nf,n)) 
        VY = np.zeros((nc//n-nf,n)) 
        p1_var = nf #just a variable I am going to use to not add vx when rejecting collisions 
    if prob == 'corr_check' :  # checking correlation
        v_arr = np.zeros((10,nc))
        v_arr[0:10,0] = np.abs(vx[0:10])
        
    print('sim... colliding particles')
    
    while cc < nc: 
        
        if prob == 4:
            kin_energy_tot = functions.tot_kin_energy(vx,vy,m)
            if kin_energy_tot < 0.1*kin_energy_start: 
                print('energy reducing reached')
                cc = nc 
                break 
            
               
        
        #plotting.particle_pos_test(x, y) #uncomment when running 3 particle test 
        
        coll = heap.heappop(collisons)
        dt = coll[0] - time 
        i = coll[1]
        j = coll[2]
        ni = coll[3]
        nj = coll[4]
      
        if j == -1 or j == -2: 
            # collide with wall          
            if ncp[i] == ni:
                
                collide_part_wall(x,y,vx,vy,r,i,j,dt,ksi)
                ncp[i] = ncp[i] + 1
                time = time + dt 
                add_new_coll(x,y,vx,vy,r,time,collisons,i,colliding_with,ncp)
                cc = cc +1
                pwcc += 1
            
                
        else: 
            #collide with particle
            if ncp[i] == ni and ncp[j] == nj:
                
                collide_part_part(x,y,vx,vy,r,m,i,j,dt,ksi) 
                ncp[i] = ncp[i] +1 
                ncp[j] = ncp[j] +1 
                colliding_with[i] = -1
                colliding_with[j] = -1
                time = time + dt 
                add_new_coll(x,y,vx,vy,r,time,collisons,i,colliding_with,ncp)
                add_new_coll(x,y,vx,vy,r,time,collisons,j,colliding_with,ncp)
                cc = cc + 1 
                ppcc += 1
            
                
         
        
        if cc%1000 == 0  : 
            print('collison number: ', cc,' collisons left:', nc-cc)
            if prob == 4: 
                 print('part of inital energy left: ',kin_energy_tot/kin_energy_start )
                
                 
        if prob == 3 and cc%100 == 0:
            time_list.append(time)
            K_all.append(functions.avrg_kin_energy(vx,vy,m))
            K1.append(functions.avrg_kin_energy(vx[:nhalf],vy[:nhalf],m[:nhalf]))
            K2.append(functions.avrg_kin_energy(vx[nhalf:],vy[nhalf:],m[nhalf:]))
        if prob == 1 and cc%n == 0: 
            if cc//n == p1_var:  
                VX[cc//n-nf-1] = vx; 
                VY[cc//n-nf-1] = vy; 
                p1_var += 1
                print('len VX',len(VX), 'cc//n-nf-1',cc//n-nf-1)
        if prob == 'corr_check': 
            v_arr[0:10,cc-1] = vx[0:10]
        
            
    print('number of part-part coll:',ppcc)
    print('number of wall-part coll:',pwcc)
             
    if prob == 1: 
        return VX,VY
    if prob == 3: 
        return  [time_list, K_all, K1, K2]
    if prob == 'corr_check':
        return v_arr
            
            
def run4(x,y,vx,vy,r,m,nc,ksi,prob): 
    print('sim... initializing collisions')
    
    colliding_with = np.zeros(len(x)) #trying this! saving which particele the collison is between 
    
    collisons = initial_collisons(x,y,vx,vy,r,colliding_with)
    
    ncp = np.zeros(len(x))  #number of collisions per particle 
    
    n = len(x)
    cc = 0 # collison count 
    time = 0 
    ppcc = 0 #particle-particle collision count
    pwcc = 0 #particle-wall collision count 
    
    kin_energy_start = functions.tot_kin_energy(vx, vy, m)

        
    print('sim... colliding particles')
    
    
    while cc < nc: 
        
        
        kin_energy_tot = functions.tot_kin_energy(vx,vy,m)
        if kin_energy_tot < 0.1*kin_energy_start: 
            print('energy reducing reached')
            cc = nc 
            break 
        coll = heap.heappop(collisons)
        dt = coll[0] - time 
        i = coll[1]
        j = coll[2]
        ni = coll[3]
        nj = coll[4] 
        vi = vx[i]**2 + vy[i]**2
        vj = vx[j]**2 + vy[j]**2
        if j == -1 or j == -2: 
            # collide with wall          
            if ncp[i] == ni:
                
                collide_part_wall(x,y,vx,vy,r,i,j,dt,ksi)
                ncp[i] = ncp[i] + 1
                time = time + dt 
                add_new_coll(x,y,vx,vy,r,time,collisons,i,colliding_with,ncp)
                cc = cc +1
                pwcc += 1               
        else: 
            #collide with particle
            if ncp[i] == ni and ncp[j] == nj:       
                collide_part_part(x,y,vx,vy,r,m,i,j,dt,ksi) 
                ncp[i] = ncp[i] +1 
                ncp[j] = ncp[j] +1 
                colliding_with[i] = -1
                colliding_with[j] = -1
                time = time + dt 
                add_new_coll(x,y,vx,vy,r,time,collisons,i,colliding_with,ncp)
                add_new_coll(x,y,vx,vy,r,time,collisons,j,colliding_with,ncp)
                cc = cc + 1 
                ppcc += 1
       
        if cc%1000 == 0  : 
            print('collison number: ', cc,' collisons left:', nc-cc)
            print('part of inital energy left: ',kin_energy_tot/kin_energy_start )

    print('number of part-part coll:',ppcc)
    print('number of wall-part coll:',pwcc)

    
    

    
    