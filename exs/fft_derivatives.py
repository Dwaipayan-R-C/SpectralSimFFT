import numpy as np
import matplotlib.pyplot as plt

L=1
def func(x):
    # f = np.sin(2*np.pi*x)**4
    # f = (x-L/2)**2
    f = np.sin(2*np.pi*x)**4 + np.cos(2*np.pi*2*x)
    return f

N_pad = 200
N_samples = 30

kappa = 2*np.pi*np.fft.fftfreq(N_samples,L/(N_samples-1)).real
k2 = np.fft.fftshift(kappa)

# Interpolation
fft_transform = np.fft.fft(func(np.arange(N_samples)/N_samples))
mid_index=N_samples//2
fft_pad = np.zeros(N_pad,dtype=complex)
fft_pad[0:mid_index]=fft_transform[0:mid_index]
fft_pad[N_pad-mid_index:]=fft_transform[N_samples-mid_index:]
fft_invs = np.fft.ifft(fft_pad)*N_pad/N_samples

# 1st order derivative with fourier
fft_derivative_1 = 1j*kappa*fft_transform
fft_derivative_pad = np.zeros(N_pad,dtype=complex)
fft_derivative_pad[0:mid_index]=fft_derivative_1[0:mid_index]
fft_derivative_pad[N_pad-mid_index:]=fft_derivative_1[N_samples-mid_index:]
fft_derivative_invs = np.fft.ifft(fft_derivative_pad,N_pad)*N_pad/N_samples

# 2nd order derivative with fourier
fft_derivative_2 = 1j * kappa * fft_derivative_1
fft_second_derivative_pad = np.zeros(N_pad,dtype=complex)
fft_second_derivative_pad[0:mid_index]=fft_derivative_2[0:mid_index]
fft_second_derivative_pad[N_pad-mid_index:]=fft_derivative_2[N_samples-mid_index:]
fft_derivative_invs_2 = np.fft.ifft(fft_second_derivative_pad,N_pad)*N_pad/N_samples

# Calculate derivative using finite differences
f_pad = func(np.arange(N_pad)/N_pad)
dfFD = np.zeros(len(f_pad),dtype=complex)
for kappa in range(N_pad-1):
    dfFD[kappa]=(f_pad[kappa+1]-f_pad[kappa])/(L/N_pad)
dfFD[-1]=dfFD[-2]

# Calculate 2nd derivative using finite difference
dfFD_2 = np.zeros(len(f_pad),dtype=complex)
dfFD_2=(np.roll(f_pad, 1) - 2*f_pad + np.roll(f_pad, -1))/(L/N_pad)**2

# Calculate FFT derivatives using Finite difference
# 1st order
kappa = 2*np.pi*np.fft.fftfreq(N_pad,L/(N_pad-1)).real
fourier_finite_derivative = np.zeros(len(f_pad),dtype=complex)
D = (np.exp((1j)*kappa*(L/N_pad))-1)/(L/N_pad)
fourier_finite_derivative=D*fft_pad
derivative_1 = np.real(np.fft.ifft(fourier_finite_derivative))*N_pad/N_samples

# Calculate second derivative using Finite diff
D = (np.exp((1j)*kappa*(L/N_pad))-2+np.exp(-(1j)*kappa*(L/N_pad)))/(L/N_pad)**2
dfFDF_2=D*fft_pad
second_derivative = np.real(np.fft.ifft(dfFDF_2))*N_pad/N_samples
fig,ax=plt.subplots(2,3,figsize=(16, 8))
fig.suptitle("Fourier transform analysis for Sin(2*np.pi*x)^4 + Cos(2*np.pi*2*x)^2", fontweight="bold")


xpad = np.arange(N_pad)/N_pad
x = np.arange(N_samples)/N_samples

# Plotting interpolation result
ax[0,0].plot(x,func(np.arange(N_samples)/N_samples),'b*')
ax[0,0].plot(np.arange(N_pad)/N_pad,np.real(fft_invs),color='r')
ax[0,0].legend(['Discrete samples','Interpolated curve'])
ax[0,0].set_title('Fourier interpolation on discrete spatial points')

# Plotting Derivative results
ax[0,1].plot(xpad,fft_derivative_invs.real, 'o',color='red')
ax[0,1].plot(xpad,dfFD.real, '-',color='b')
ax[0,1].legend(['Fourier derivative', 'Finite difference'])
ax[0,1].set_title('Fourier derivative vs Finite difference derivative')

# Plot 2nd derivative
ax[0,2].plot(xpad, fft_derivative_invs_2.real,'o',color='red')
ax[0,2].plot(xpad, dfFD_2, '-',color='b')
ax[0,2].legend(['Fourier second derivative', 'Finite second derivative'])
ax[0,2].set_title("Second derivative plot")

# Plot Fourier derivative using Finite difference
ax[1,0].plot(xpad,np.real(derivative_1),'o',color='red')
ax[1,0].plot(xpad,np.real(dfFD), '-',color='b')
ax[1,0].legend(['Fourier derivative', 'Finite derivative'])
ax[1,0].set_title("Plots of Fourier derivative using Finite difference")

ax[1,1].plot(xpad, second_derivative.real,'o',color='red')
ax[1,1].plot(xpad,np.real(dfFD_2), '-',color='b')
ax[1,1].legend(['Fourier second derivative', 'Finite second derivative'])
ax[1,1].set_title("Plots of Fourier second derivative using Finite difference")

ax[1,2].set_title("Fourier derivative using finite difference (Reciprocal space)")
ax[1,2].plot(xpad,fourier_finite_derivative.real,color='red')

fig.tight_layout()
# fig.savefig('plots//exercise_3//Fourier_derivatives_2.png')
plt.show()