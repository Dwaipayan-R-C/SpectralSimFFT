import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np

#  Function definition for grids
def second_derivative(T, dx):
    return (np.roll(T, 1) - 2*T + np.roll(T, -1))/dx**2

def time_marching(T, alpha, sources, dx, dt):
    return T + (alpha*second_derivative(T, dx) + sources)*dt


# Spatial grid and the time increment
nx = 1000 # Number of grid points
dx = 0.1 # Grid spacing (delta x in the above equations)
nt = 10000 # Number of time integration steps
dt = 0.001 # Time step (delta t in the above equations)
alpha = 1 # Transport coefficient 

# Initial condition: Gaussian temperature profile
x = np.linspace(0, (nx-1)*dx, nx)
L = (nx-1)*dx-0
sigma0 = 40*dx
mu = nx*dx/2
T = np.exp(-(x-mu)**2/(2*sigma0**2))/(np.sqrt(2*np.pi)*sigma0)
nt *= 10        # We'll run this five times as long as the above propagation simulation

# Heat sources
sources = np.zeros_like(T)
# sources[nx//4] = 1
# sources[3*nx//4] = -0.5
# sources[7*nx//8] = -0.5


# Fourier definition of Fundamental frequency
u_hat = np.fft.fft(T).real
kappa = 2*np.pi*np.fft.fftfreq(nx,L/(nx-1)).real
# source_T = np.fft.fft(sources).real
# source_T = np.fft.fftshift(source_T)
# Plot info
fig,ax = plt.subplots(1,2, figsize=(15, 8))
t=0
# Time propagation
for t in range(nt):
# while(u_hat[t]!=u_hat[t-1]):
    t+=1
    T = time_marching(T, alpha, sources, dx, dt)
    
    # u_hat(k,t+del_t) = -D*K**2*u_hat(k,t)
    u_hat[:] = u_hat[:]*(1-alpha*dt*kappa**2) 
    # u_hat[:] = u_hat[:]*(1-alpha*dt*kappa**2) + source_T*dt
    
    if t % (nt // 10) == 0:
        ax[0].plot(x, T, '-', label=f'$t={t*dt}$')
        ax[0].set_xlabel('Position $x$')
        ax[0].set_ylabel('Temperature distribution')
        ax[0].set_title('Heat diffusion using Finite difference')
        u_fft = np.fft.ifft(u_hat).real
        ax[1].plot(x,u_fft)
        ax[1].set_xlabel('Position $x$')
        ax[1].set_ylabel('Temperature distribution')
        ax[1].set_title('Heat diffusion using Spectral method (Fourier)')
        
        
fig.tight_layout()
# fig.savefig('plots//exercise_4//Heat_transport.png')
plt.show()
