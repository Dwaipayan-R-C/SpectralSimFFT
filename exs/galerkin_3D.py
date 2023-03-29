import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
import itertools

N       = 2          # number of basis functions
L       = 1           # length of x and y
L_steps = 30         # number of grid points


# Spatial grid and the time increment
nx      = L_steps # Number of grid points in x direction
ny      = L_steps # Number of grid points in y direction
nz      = L_steps # Number of grid points in z direction
dx      = L/L_steps # Grid spacing (delta x)
dy      = L/L_steps # Grid spacing (delta y)
dz      = L/L_steps # Grid spacing (delta z)
alpha   = 1 # Transport coefficient


# heat sources and sinks:
sources = np.zeros((nx,ny,nz))
#sources = np.ones_like(T)
sources[nx//4, ny//4, nz//4]       =  0.5
sources[3*nx//4, 3*ny//4, 3*nz//4] = -0.3

# Ex04-A
qs = sources

x = np.arange(L_steps)*dx
y = np.arange(L_steps)*dy
z = np.arange(L_steps)*dz

T_N = np.zeros((nx, ny, nz))

# boundary conditions are T(0) = 0 and T(L) = 0
# so use the basis function x**k*(x - L), with k = 0..N-1
v = np.zeros((N, N, N, nx, ny, nz))
for ix in range(nx): # go over x
    for iy in range(ny): # go over y
        for iz in range(nz): # go over z
            for k in range(N):
                for l in range(N):
                    for m in range(N):
                        v[k, l, m, ix, iy, iz] = np.sin(np.pi*(k+1)/L * x[ix]) * np.sin(np.pi*(l+1)/L * y[iy]) * np.sin(np.pi*(m+1)/L * z[iz])



k = 2*np.pi*np.fft.fftfreq(L_steps,L/(L_steps-1)).real
fft = np.fft.fftn(v)
ffts = 1j*k*fft
fftss = 1j*k*ffts

vs = np.fft.ifftn(ffts).real
vss = np.fft.ifftn(fftss).real


A = np.zeros((N*N*N,N*N*N))
for ki, kj, li, lj, mi, mj in itertools.product(range(N), repeat=6):
    row_idx = N*N*ki + N*li + mi
    col_idx = N*N*kj + N*lj + mj
    A[row_idx, col_idx] = np.sum(vss[kj, lj, mj] * v[ki, li, mi])


b = np.sum(-qs * v, axis=(3, 4, 5)).reshape(N*N*N)

# calculate T~k:
c = solve(A,b)

#region Plots
T_N = np.sum(c.reshape(N, N, N, 1, 1, 1) * v, axis=(0, 1, 2))

fig = plt.figure(figsize=(12, 8))
X, Y = np.meshgrid(x, y)


ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.set_title('z=0')
print(0)
z_val = 0
ax.scatter3D(X, Y, T_N[:,:,z_val], c=T_N[:,:,z_val])


ax = fig.add_subplot(2, 3, 2, projection='3d', sharex=ax, sharey=ax)
ax.set_title('z=1*nz//4')
print(1)
z_val = 1*nz//4
ax.scatter3D(X, Y, T_N[:,:,z_val], c=T_N[:,:,z_val])


ax = fig.add_subplot(2, 3, 3, projection='3d', sharex=ax, sharey=ax)
ax.set_title('z=2*nz//4')
print(2)
z_val = 2*nz//4
ax.scatter3D(X, Y, T_N[:,:,z_val], c=T_N[:,:,z_val])


ax = fig.add_subplot(2, 3, 4, projection='3d', sharex=ax, sharey=ax)
ax.set_title('z=3*nz//4')
print(3)
z_val = 3*nz//4
ax.scatter3D(X, Y, T_N[:,:,z_val], c=T_N[:,:,z_val])


ax = fig.add_subplot(2, 3, 5, projection='3d', sharex=ax, sharey=ax)
ax.set_title('z=4*nz//4-1')
print(4)
z_val = 4*nz//4-1
ax.scatter3D(X, Y, T_N[:,:,z_val], c=T_N[:,:,z_val])

ax = fig.add_subplot(2, 3, 6, projection='3d', sharex=ax, sharey=ax)
ax.set_title('z=nz-1')
print(5)
z_val = nz-1
ax.scatter3D(X, Y, T_N[:,:,z_val], c=T_N[:,:,z_val])
fig.tight_layout()

#endregion 

# plt.savefig('plots/galerkin_2D.png')
plt.show()

