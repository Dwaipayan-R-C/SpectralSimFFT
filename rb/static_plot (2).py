import numpy as np
import matplotlib.pyplot as plt

# Define constants
L = 100
n = 1
theta_0 = 1

def w_n(n, R, k):
    return R * k**2 / (-n**2 * np.pi**2 - k**2)**2

def w(z, n, R, k):
    return w_n(n, R, k) * np.sin(np.pi * n * z)

def theta(z, n):
    return theta_0 * np.sin(np.pi * n * z)

# TASK 1
# Calculate the critical Rayleigh number and wavenumbers
k_values = np.linspace(0.8, 7, 30)
Rc_values = np.zeros_like(k_values)
for i, k in enumerate(k_values):
    Rc = (np.pi**2 + k**2)**3 / k**2
    Rc_values[i] = Rc

# Find the minimum Rc value and the corresponding wavenumber
k_critical = k_values[np.argmin(Rc_values)]
Rc_critical = np.min(Rc_values)
print("Critical Rayleigh number: ", Rc_critical)
print("Critical wavenumber: ", k_critical)

# TASK 2
# The wavevectors
k_x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(L, L / (L - 1)))
k_y = k_x.copy()
KX, KY = np.meshgrid(k_x, k_y)
#
x = np.linspace(0, L, L)
y = np.linspace(0, L, L)
z = np.linspace(0, 1, L)
X, Y = np.meshgrid(x,y)
XZ,Z = np.meshgrid(x,z)
kx_critical =  2 * np.pi / L * k_critical
kz_critical = 2 * np.pi / L * n

U = -np.real(w(Z, n, Rc_critical, kz_critical) * np.sin(kx_critical * XZ))
#V = np.real(w(Z, n, Rc_critical, kz_critical) * np.cos(kx_critical * X))
W = np.real(w(Z, n, Rc_critical, kz_critical) * np.cos(kx_critical * XZ))
Theta = np.real(theta(0.5, n) * np.cos(kx_critical * X))

#%matplotlib notebook
fig, (axW, axT) = plt.subplots(2,1,figsize=(4, 4))
fig.tight_layout()
stream = axW.streamplot(XZ[:,0:40], Z[:,0:40], U[:,0:40], W[:,0:40], color='k', linewidth=0.5, density=1.)#, arrowstyle='->', arrowsize=0.5)
temp = axT.contourf(X[:,0:40], Y[:,0:40], Theta[:,0:40], cmap='coolwarm', levels=np.linspace(-1, 1, 21))
#cbar1 = plt.colorbar(temp, ax=axW)
#cbar1.ax.set_ylabel('Temperatur
#ax.set_title('Convective pattern at z=1/2')
axW.set_xlabel('x')
axW.set_ylabel('z')
axT.set_xlabel('x')
axT.set_ylabel('y')
plt.show()