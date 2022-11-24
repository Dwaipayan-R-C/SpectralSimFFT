import numpy as np
import matplotlib.pyplot as plt

# odd dimension for simplicity
n    = 10
npad = 300
duration = 2*np.pi
x    = np.arange(0., duration, duration/n)
xpad = np.arange(0., duration, duration/npad)

f = np.sin(x) 
f_fwd = np.fft.fft(f)
f_fwd_pad = np.zeros(npad,dtype=complex)
analyticdF = np.cos(2*np.pi*xpad)
h = n - 3
f_fwd_pad[0:h+1]   = f_fwd[0:h+1]
f_fwd_pad[npad-h:] = f_fwd[n-h:]
f_interpolated = np.fft.ifft(f_fwd_pad)*npad/n

# Calculate derivative using finite differences
dfFD = np.zeros(len(f_fwd_pad),dtype=complex)
for kappa in range(npad-1):
    dfFD[kappa]=(f_fwd_pad[kappa+1]-f_fwd_pad[kappa])/(duration/npad)
dfFD[-1]=dfFD[-2]

# Calculating Fourier derivative 
kappa = (2*np.pi/duration)*np.arange(0,npad)
kappa = np.fft.fftshift(kappa)
dfhat = kappa*f_fwd_pad*(1j)
dfFFT = np.real(np.fft.ifft(dfhat))

plt.figure(1)
plt.plot(x,f,'b*')
plt.plot(xpad,np.real(f_interpolated),color='r')
plt.legend(['Discrete samples','Interpolated curve'])
plt.title('Fourier interpolation on discrete spatial points')


plt.figure(2)
# plt.plot(xpad,analyticdF, '--',color='b')
plt.plot(xpad,dfFFT.real, color='red')
plt.legend(['Finite difference','Fourier derivative'])
plt.title('Fourier derivative and interpolation on discrete spatial points')

plt.show()

