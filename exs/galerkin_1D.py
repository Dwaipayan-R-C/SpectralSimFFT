import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

L = 4
N = 6
divs = 100
x = np.linspace(0,L,divs)
h = x[1]-x[0]
A = np.zeros((N,N))
B=np.zeros((N,1))
alpha = 1

def f(x):
    sources = np.zeros_like(x)
    sources[:] = 1
    return sources
  
def psi(x,i0):    
    return x**i0*(x-L)
    
def psidd(x,i0):
    N = len(x)
    kappa = 2*np.pi*np.fft.fftfreq(N,L/(N)).real
    fft_transform = np.fft.fft(psi(x,i0))
    D = (np.exp((1j)*kappa*(L/N))-2+np.exp(-(1j)*kappa*(L/N)))/(L/N)**2
    dfFDF_2=((1j)*kappa)**2*fft_transform    
    second_derivative = np.real(np.fft.ifft(dfFDF_2))
    return second_derivative

def As(x,i0,j0):
    dx = x[1]-x[0]
    dot_value = np.dot(psi(x,i0),psidd(x,j0))
    integrated_value = np.sum(dot_value*dx)
    return integrated_value

def bs(x,i0):
    dx = x[1]-x[0]
    dot_value = np.dot(psi(x,i0),-f(x))
    integrated_value = np.sum(dot_value*dx)
    return integrated_value
    
for i1 in range(0,N):           
    for j1 in range(0,N):
        A[i1,j1] = As(x,i1,j1)
    B[i1]=bs(x,i1)

C = spsolve(A, B)

ps = np.zeros((divs,N))
psdd = np.zeros((divs,N))

for i2 in range(0,N):
    ps[:,i2] = psi(x,i2)
    psdd[:,i2] = psidd(x,i2)

# psdd[np.isnan(psdd)]=psdd[2,1]
U = np.matmul(ps,C)

# Using Finite difference
sources = np.zeros_like(x)
sources[:] = 1
A = coo_matrix((-2*np.ones(divs), (np.arange(divs), np.arange(divs))), shape=(divs, divs))
A += coo_matrix((np.ones(divs-1), (np.arange(divs-1)+1, np.arange(divs-1))), shape=(divs, divs))
A += coo_matrix((np.ones(divs-1), (np.arange(divs-1), np.arange(divs-1)+1)), shape=(divs, divs)) 
b = -h**2/alpha * sources
T_FD = spsolve(A, b)

# Analytical
T_analytical = sources/2*(x*L-x**2)

residual = np.matmul(psdd,C) + f(x)
error = U - T_analytical
fig,ax=plt.subplots(1,2,figsize=(14, 6))
ax[0].plot(x,T_analytical,'-',linewidth=4.0,color='blue',label='Analytical Solution')
ax[0].plot(x,T_FD,'-',linewidth=4.0,color='red',label='Finite difference Solution')
ax[0].plot(x, U,  '-',linewidth=4.0,color='green',label='Numerical Solution')
ax[0].legend(['Analytical Solution','Finite difference Solution','Galerkin Spectral Solution'])
ax[0].set_xlabel('Position X')
ax[0].set_ylabel('Temperature (T)')

ax[1].plot(x[0:98],residual[0:98],'-',linewidth=4.0,color='blue')
ax[1].plot(x,error,'-',linewidth=4.0,color='red')
ax[1].legend(['Residual','Error'])
ax[1].set_xlabel('Position X')
fig.suptitle("PDE: u'' + Qs/alpha = 0")
fig.tight_layout()
# fig.savefig('plots//exercise_4//Galerkin_heat_diffusion.png')
plt.show()
print()
