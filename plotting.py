# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt 


def particle_pos(x,y,r,title='in sim'):
    plt.figure(figsize= (10,10))
    plt.scatter(x,y,s=20)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.hlines(0, 0, 1)
    plt.hlines(1, 0, 1)
    plt.vlines(0, 0, 1)
    plt.vlines(1, 0, 1)
    plt.title(title)
    plt.show()
    
def particle_pos_p4(x,y):
    plt.figure(figsize=(10,10))
    plt.scatter(x[1:],y[1:],s=20)
    plt.scatter(x[0] ,y[0] ,s=100)
    plt.hlines(0, 0, 1)
    plt.hlines(1, 0, 1)
    plt.vlines(0, 0, 1)
    plt.vlines(1, 0, 1)
    plt.show()

def speed_histogram(vx,vy):
    v_abs = np.sqrt(vx**2+vy**2)
    v_a = v_abs.flatten()
    plt.figure()
    plt.hist(v_a,101,(0,2),density=True)
    plt.xlabel('v')
    plt.show()
    
def speed_histogram2(vx1,vy1,vx2,vy2):
    v1_abs = np.sqrt(vx1**2+vy1**2)
    v2_abs = np.sqrt(vx2**2+vy2**2)
    v1a = v1_abs.flatten()
    v2a = v2_abs.flatten()
    
    plt.figure(figsize =(10,15))
    plt.subplot(2,1,1)
    
    plt.hist(v1_abs,41,(0,4),density=True)
    plt.subplot(2,1,2)
    plt.hist(v2_abs,41,(0,4),density=True)
    #plt.xlabel('v')
    plt.show()
    
def compare_to_Boltzmann(vx,vy,pref):
    
    v_axis = np.linspace(0,5,300)
    v_abs = np.sqrt(vx**2+vy**2)
    v = v_abs.flatten()
    
    rho = pref * v_axis * np.exp(- pref/2 *v_axis**2 )
    plt.figure()
    plt.plot(v_axis, rho)
    plt.hist(v,101 ,(0,5),density=True)
    plt.show()
    
def kinetic_energy_aft(p3data): 
    plt.rcParams.update({'font.size': 12})
    time = p3data[0]
    ak1 = p3data[2]
    ak2 = p3data[3]
    aktot = p3data[1]
    
    # fig, ax = plt.subplots(3,figsize=(5,9))
    # ax[0].plot(time,ak1)
    # ax[0].set_ylim(0,2)
    # ax[1].plot(time,ak2)
    # ax[1].set_ylim(0,2)
    # ax[2].plot(time,aktot)
    # ax[2].set_ylim(0,2)
    plt.figure(figsize=(8,4))
    plt.plot(time,aktot, label ='all particles')
    plt.plot(time,ak1,   label ='particle type 1')
    plt.plot(time,ak2,   label ='particle type 2')
    plt.ylim(0,2.5)
    plt.legend()
    plt.show()
    

    
    

#just testing with 3 particles 
def particle_pos_test(x,y,title='in sim'):
    # for figsize(10,10), r = 0.05 it fits with s = 1800
    plt.figure(figsize= (10,10))
    plt.hlines(0, 0, 1)
    plt.hlines(1, 0, 1)
    plt.vlines(0, 0, 1)
    plt.vlines(1, 0, 1)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.scatter(x[0],y[0],s=1800,label='0')
    plt.scatter(x[1],y[1],s=1800,label='1')
    plt.scatter(x[2],y[2],s=1800,label='2')
    plt.legend()
    plt.title(title)
    plt.show()
