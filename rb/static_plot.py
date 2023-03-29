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

def RaRealistic(k):
    return (28*k**6+952*k**4+20832*k**2+141120)/(27*k**2)
# TASK 1
# Calculate the critical Rayleigh number and wavenumbers
k_values = np.linspace(0.8, 7, 100)
Rc_values = np.zeros_like(k_values)
for i, k in enumerate(k_values):
    Rc = (np.pi**2 + k**2)**3 / k**2
    Rc_values[i] = Rc
n_Ra = RaRealistic(k_values)
# Find the minimum Rc value and the corresponding wavenumber
k_critical = k_values[np.argmin(Rc_values)]
Rc_critical = np.min(Rc_values)

plt.xlabel('Wave vector K')
plt.ylabel('Rayleigh numbers')
plt.plot(k_values,Rc_values,color='red')
plt.plot(k_values,n_Ra,color='black')
#print("Critical Rayleigh number: ", Rc_critical)
#print("Critical wavenumber: ", k_critical)
k_m = k_values[np.argmin(Rc_values)+1]
Rm = Rc_values[np.argmin(Rc_values)+1]
plt.ylim(0,5000)
plt.scatter(k_m,Rm,color='black',alpha=.9)
plt.text(4,3500,f'km={np.round(k_m,3)},\n Rc={np.round(Rm,2)} ' ,ha='center', va='center',fontsize=10,  bbox=dict(facecolor='red', alpha=0.5) )
plt.legend(["Linear","Non-linear"])
plt.title('Critical Rayleigh')
plt.tight_layout()
# plt.show()
# TASK 2
# The wavevectors

k_x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(L, L / (L - 1)))
k_y = k_x.copy()
KX, KY = np.meshgrid(k_x, k_y)

x = np.linspace(0, L, L)
y = np.linspace(0, L, L)
z = np.linspace(0, 1, L)
X, Y = np.meshgrid(x,y)
XZ,Z = np.meshgrid(x,z)
kx_critical = 2 * np.pi / L * k_m
kz_critical = 2 * np.pi / L * n

U = -np.real(w(Z, n, Rm, kz_critical) * np.sin(kx_critical * X))
V = np.real(w(Z, n, Rm, kz_critical) * np.sin(kx_critical * Y))
W = np.real(w(Z, n, Rm, kz_critical) * np.cos(kx_critical * X))
Theta = np.real(theta(0.5, n) * np.cos(kx_critical * X))
Theta_z = np.real(theta(z, n) * np.cos(kx_critical * XZ))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 6))

stream = ax1.streamplot(X[:,0:45], Y[:,0:45], U[:,0:45], V[:,0:45], color='k', linewidth=0.5, density=1.)#, arrowstyle='->', arrowsize=0.5)
temp = ax2.contourf(X[:,0:45], Y[:,0:45], Theta[:,0:45], cmap='inferno', levels=np.linspace(-1, 1, 50))
# temp = ax2.contourf(X[:,0:23], Y[:,0:23], Theta[:,0:23], cmap='inferno', levels=np.linspace(-1, 1, 21))
color=plt.colorbar(temp)
fig.suptitle('Convective pattern at XZ plane')
ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax2.set_xlabel('x')
ax2.set_ylabel('z')
fig.tight_layout()
plt.show()