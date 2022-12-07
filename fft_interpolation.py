import numpy as np
import matplotlib.pyplot as plt

# odd dimension for simplicity
n    = 50                                           # Discrete samples
npad = 500                                          # Padding size
duration = 2*np.pi                                  # L (Total length)
x    = np.arange(0., duration, duration/n)          # Sample array (spatial)
xpad = np.arange(0., duration, duration/npad)       # Padding array

# Function definition
f = 3*np.sin(2*x) 
f_pad = 3*np.sin(2*xpad) 

# FFT transformation & interpolation
f_fwd = np.fft.fft(f)
f_fwd_pad = np.zeros(npad,dtype=complex)
analyticdF = np.cos(xpad)

# Padded like 7,9,5,2,0,0,0,0,6,8,1,3
h = n - 3       # 7
f_fwd_pad[0:h+1]   = f_fwd[0:h+1]
f_fwd_pad[npad-h:] = f_fwd[n-h:]
f_interpolated = np.fft.ifft(f_fwd_pad)*npad/n

# Calculate derivative using finite differences
dfFD = np.zeros(len(f_pad),dtype=complex)
for kappa in range(npad-1):
    dfFD[kappa]=(f_pad[kappa+1]-f_pad[kappa])/(duration/npad)
dfFD[-1]=dfFD[-2]

# Calculate 2nd derivative using finite difference
dfFD_2 = np.zeros(len(f_pad),dtype=complex)
dfFD_2=(np.roll(f_pad, 1) - 2*f_pad + np.roll(f_pad, -1))/(duration/npad)**2


# Calculating Fourier derivative 
    # k = np.arange(0,npad)
    # Fundametal frequency = (2*np.pi/duration)
    # kappa = k*Fundametal frequency
kappa = (2*np.pi/duration)*np.arange(0,npad)
kappa = np.fft.fftshift(kappa)
dfhat = kappa*f_fwd_pad*(1j)
dfFFT = np.real(np.fft.ifft(dfhat))*npad/n


# Calculate fourier derivative using finite difference
kappa = (2*np.pi/duration)*np.arange(0,npad)
dfFDF = np.zeros(len(f_pad),dtype=complex)
D = (np.exp((1j)*kappa*(duration/npad))-1)/(duration/npad)
dfFDF=D*f_fwd_pad
derivative = np.real(np.fft.ifft(dfFDF))*npad/n


# Calculating second derivative by Fourier
kappa = (2*np.pi/duration)*np.arange(0,npad)
kappa = np.fft.fftshift(kappa)
dfft_2derv = kappa*(1j)*dfFDF
dfft_2derv = np.real(np.fft.ifft(dfft_2derv))*npad/n

# Calculaate second derivative using Finite diff
kappa = (2*np.pi/duration)*np.arange(0,npad)
D = (np.exp((1j)*kappa*(duration/npad))-2+np.exp(-(1j)*kappa*(duration/npad)))/(duration/npad)**2
dfFDF_2=D*f_fwd_pad
second_derivative = np.real(np.fft.ifft(dfFDF_2))*npad/n


fig,ax=plt.subplots(2,3,figsize=(16, 8))
fig.suptitle("Fourier transform analysis for 3*Sin(2x)", fontweight="bold")

# Plotting interpolation result
ax[0,0].plot(x,f,'b*')
ax[0,0].plot(xpad,np.real(f_interpolated),color='r')
ax[0,0].legend(['Discrete samples','Interpolated curve'])
ax[0,0].set_title('Fourier interpolation on discrete spatial points')

# Plotting Derivative results
ax[0,1].plot(xpad,dfFFT.real, 'o',color='red')
ax[0,1].plot(xpad,dfFD.real, '-',color='b')
ax[0,1].legend(['Fourier derivative', 'Finite difference'])
ax[0,1].set_title('Fourier derivative vs Finite difference derivative')


# ax[1,1].plot(xpad,dfhat.real)
# ax[1,1].set_title("Fourier derivative (Reciprocal space)")

# Plot Fourier derivative using Finite diff in Reciprocal space
# We see the difference because Kappa is not shifted in this method
# This method works because We don't use any 
ax[1,2].set_title("Fourier derivative using finite difference (Reciprocal space)")
ax[1,2].plot(xpad,dfFDF.real)

# Plot Fourier derivative using Finite difference
ax[1,0].plot(xpad,np.real(derivative),'o',color='red')
ax[1,0].plot(xpad,np.real(dfFD), '-',color='b')
ax[1,0].legend(['Fourier derivative', 'Finite derivative'])
ax[1,0].set_title("Plots of Fourier derivative using Finite difference")

# Plot 2nd deivative
ax[0,2].plot(xpad, dfft_2derv.real,'o',color='red')
ax[0,2].plot(xpad, dfFD_2.real, '-',color='b')
ax[0,2].legend(['Fourier second derivative', 'Finite second derivative'])
ax[0,2].set_title("Second derivative plot")


ax[1,1].plot(xpad, second_derivative.real)
ax[1,1].set_title("Plots of Fourier second derivative using Finite difference")
fig.tight_layout()
fig.savefig('plots//exercise_3//Fourier_derivatives.png')
plt.show()


