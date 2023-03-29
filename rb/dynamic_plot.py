import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import PillowWriter
from matplotlib import colors

# Define constants
Lx         = 20
Ly         = 20
Lz         = 1
theta_0    = 1
time_steps = 1000
dt         = 0.001
Nx         = 101
Ny         = 101
Nz         = 21
n          = 1

# Calculate the critical Rayleigh number and wavenumbers
k_values = np.linspace(0.8, 10, 20)
Rc_values = np.zeros_like(k_values)
for i, k in enumerate(k_values):
    #Rc = (np.pi**2 + k**2)**3 / k**2
    Rc = (28*k**6+952*k**4+20832*k**2+141120)/(27*k**2)
    Rc_values[i] = Rc

# Find the minimum Rc value and the corresponding wavenumber
k_critical = k_values[np.argmin(Rc_values)]
Rc_critical = np.min(Rc_values)
print("Critical Rayleigh number: ", Rc_critical)
print("Critical wavenumber: ", k_critical)
R = Rc_critical


# set R to a fixed value:
R = 10000

def solve_coupled_equations(theta_kx_ky, u_kx_ky, v_kx_ky, w_kx_ky, dt):
    kx = 2 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, Ly / Ny)
    
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    # Calculate real space values
    theta_real_space = np.fft.ifftshift(theta_kx_ky).real
    u_real_space = np.fft.ifft2(u_kx_ky).real
    v_real_space = np.fft.ifft2(v_kx_ky).real
    w_real_space = np.fft.ifft2(w_kx_ky).real

    # Calculate the gradient of theta in real space
    # grad_theta_x = np.roll(theta_real_space, -1, axis=0) - np.roll(theta_real_space, 1, axis=0)
    # grad_theta_y = np.roll(theta_real_space, -1, axis=1) - np.roll(theta_real_space, 1, axis=1)
    # Calculate the gradient of theta in x and y directions
    grad_theta_x, grad_theta_y = np.gradient(theta_real_space)

    # Calculate N in x and y directions
    N_real_space = u_real_space * grad_theta_x + v_real_space * grad_theta_y

    # Calculate N in z direction
    N_z_real_space = w_real_space * np.fft.ifft2(-1j * KX * theta_kx_ky).real

    # Combine N_real_space and N_z_real_space to get the whole N(x,y,z,t)
    N_total_real_space = N_real_space + N_z_real_space

    # Transform N_total_real_space to Fourier space
    N_kx_ky = np.fft.fft2(N_total_real_space)

    # Update Fourier coefficients using the given equations from project 3 "Equations of motion"
    theta_dot_kx_ky = -(28/3 + K2) * theta_kx_ky - (7/63) * w_kx_ky - 1/140*N_kx_ky
    # Calculate theta_kx_ky
    theta_kx_ky += dt * theta_dot_kx_ky
    
    # Calculate u_kx_ky, v_kx_ky, and w_kx_ky
    # Find the K's which don't produce a division by zero or huge values and only use them to calculate u_kx_ky, v_kx_ky and w_kx_ky
    k_mask = (K2**2 > 1e-15) & (K2 < 1e3)
        
    # Calculate the x component of the velocity field (see Project 3 for the formulars: The x-component of the fluid velocity  u(r,t))
    u_kx_ky[k_mask] = (R * 1j * KX[k_mask] * theta_kx_ky[k_mask]) / ((K2[k_mask]**2) / 2 + 12 * K2[k_mask])
  
    # Calculate the y component of the velocity field (see Project 3 for the formulars: The y-component of the fluid velocity  v(r,t))
    v_kx_ky[k_mask] = (R * 1j * KY[k_mask] * theta_kx_ky[k_mask]) / ((K2[k_mask]**2) / 2 + 12 * K2[k_mask])
    
    # Calculate the z component of the velocity field
    # derivation based on formulars of project 3 (equations of motion)
    w_kx_ky[k_mask] = -((R*K2[k_mask] * theta_kx_ky[k_mask])/30)/(((K2[k_mask]**2) / 140) + (2 * K2[k_mask] / 15) + 4 )
    
    return theta_kx_ky, u_kx_ky, v_kx_ky, w_kx_ky

# Placeholder arrays for Fourier coefficients
theta_kx_ky = np.zeros((Nx, Ny), dtype=np.complex128)
u_kx_ky = np.zeros((Nx, Ny), dtype=np.complex128)
v_kx_ky = np.zeros((Nx, Ny), dtype=np.complex128)
w_kx_ky = np.zeros((Nx, Ny), dtype=np.complex128)

# Random Initialization
theta_kx_ky = np.fft.fftshift(np.fft.fft2(np.random.rand(Nx, Ny)))

# Define the update function for the animation
def update(t):
    print(t)
    global theta_kx_ky, u_kx_ky, v_kx_ky, w_kx_ky, dt, stream, im, im2, im_u, im_v, im_w
    theta_kx_ky, u_kx_ky, v_kx_ky, w_kx_ky = solve_coupled_equations(theta_kx_ky, u_kx_ky, v_kx_ky, w_kx_ky, dt)

    # Get the Temperature field in real space
    plot_data = np.fft.ifft2(theta_kx_ky).real
    plot_data_k = theta_kx_ky.real
    im.set_data(plot_data)
    im.autoscale()
    im2.set_data(np.fft.fftshift(plot_data_k))
    im2.autoscale()

    # Get the velocity field in real space
    u_data = np.fft.ifft2(u_kx_ky).real
    v_data = np.fft.ifft2(v_kx_ky).real
    w_data = np.fft.ifft2(w_kx_ky).real
    
    # Update the Plot of the velocity field
    im_u.set_data(u_data)
    im_u.autoscale()

    im_v.set_data(v_data)
    im_v.autoscale()

    im_w.set_data(w_data)
    im_w.autoscale()
    

# Set up the plot
fig, ((ax_u, ax_v, ax_w, ax_theta, ax_theta_k)) = plt.subplots(1,5,figsize=(20, 4))
fig.tight_layout(pad=3, h_pad=5, w_pad=5)

ax_theta.set_title('Theta(r,t)')
ax_theta.set_xlabel('x')
ax_theta.set_ylabel('y')
ax_theta_k.set_title('Theta(kx,ky)')
ax_theta_k.set_xlabel('kx')
ax_theta_k.set_ylabel('ky')
ax_u.set_title('u(r,t)')
ax_u.set_xlabel('x')
ax_u.set_ylabel('y')
ax_v.set_title('v(r,t)')
ax_v.set_xlabel('x')
ax_v.set_ylabel('y')
ax_w.set_title('w(r,t)')
ax_w.set_xlabel('x')
ax_w.set_ylabel('y')

plot_data = np.fft.ifft2(theta_kx_ky).real
plot_data_k = theta_kx_ky.real
im = ax_theta.imshow(plot_data, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')
im2 = ax_theta_k.imshow(plot_data_k, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')

plt.colorbar(im, ax=ax_theta)

u_data = np.fft.ifft2(u_kx_ky).real
v_data = np.fft.ifft2(v_kx_ky).real
w_data = np.fft.ifft2(w_kx_ky).real
im_u = ax_u.imshow(u_data, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')
im_v = ax_v.imshow(v_data, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')
im_w = ax_w.imshow(w_data, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')

# Create the animation
ani = FuncAnimation(fig, update, frames=range(time_steps), interval=1, repeat=False, cache_frame_data=True) #, repeat=False, blit=True, )
#writer = PillowWriter(fps=24)  # Passen Sie die FPS (Frames pro Sekunde) nach Bedarf an
#ani.save("my_animation.gif", writer=writer)
plt.show()
