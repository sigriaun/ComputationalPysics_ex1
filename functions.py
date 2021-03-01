import numpy as np 

def avrg_speed(vx,vy): 
    v = np.sqrt(vx**2 + vy**2)
    return np.mean(v)


def avrg_kin_energy(vx,vy,m): 
    v_sq = vx**2 + vy**2 
    K = 0.5 * m * v_sq
    return np.mean(K)

def crater_size(x_ini,y_ini,x,y):
    n = len(x)
    x_unchanged = x_ini == x
    y_unchanged = y_ini == y

    unchanged = x_unchanged*y_unchanged

    n_c = np.count_nonzero(unchanged)
    return (n-n_c)/n

def tot_kin_energy(vx,vy,m):
    Ek = 0.5*(vx**2+vy**2)*m
    return np.sum(Ek)
