import numpy as np
import _main_ as sim

def Task1():
    low,high = 0.7,7
    plot_bool = True
    sim.RayleighK(low,high,plot_bool)
    
def Task2():
    length = 100
    n=1
    θn=1
    low,high = 0.7,7
    plot_bool = True
    sim.StaticSim(length,n,θn,low,high,plot_bool)  
    
def Task3():
    # Variabe definition
    Ra=3000
    Nx=100
    Ny=100
    Nz=50
    Lx=30
    Ly=30
    Lz=1
    n=1
    timescale=1000
    θn=1
    dt=0.001
    save_every=50  
    temp_sim=True
    
    # Plotting condition
    if(temp_sim==True):
        vel_sim=False
    else:
        vel_sim=True
    sim.DynamicSim(Ra,Nx,Ny,Nz,Lx,Ly,Lz,n,timescale,θn,dt,save_every,temp_sim,vel_sim)


# Run simulation
# Task1()
# Task2()
Task3()