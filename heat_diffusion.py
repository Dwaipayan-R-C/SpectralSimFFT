import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np


def second_derivative(T, dx):
    return (np.roll(T, 1) - 2*T + np.roll(T, -1))/dx**2

def time_marching(T, alpha, sources, dx, dt):
    return T + (alpha*second_derivative(T, dx) + sources)*dt


### Spatial grid and the time increment
nx = 1000 # Number of grid points
dx = 0.1 # Grid spacing (delta x in the above equations)
nt = 10000 # Number of time integration steps
dt = 0.001 # Time step (delta t in the above equations)
alpha = 1 # Transport coefficient 

### Initial condition: Gaussian temperature profile
x = np.linspace(0, (nx-1)*dx, nx)
sigma0 = 40*dx
mu = nx*dx/2
T = np.exp(-(x-mu)**2/(2*sigma0**2))/(np.sqrt(2*np.pi)*sigma0)

nt *= 10 # We'll run this five times as long as the above propagation simulation
sources = np.zeros_like(T)

for t in range(nt):
    T = time_marching(T, alpha, sources, dx, dt)
    if t % (nt // 20) == 0:
        plt.plot(x, T, '-', label=f'$t={t*dt}$')
plt.figure(1)
plt.legend(loc='best')
plt.xlabel('Position $x$')
plt.ylabel('Temperature $T$')
plt.show()
