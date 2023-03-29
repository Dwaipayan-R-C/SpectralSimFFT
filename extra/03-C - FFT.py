import numpy as np
import matplotlib.pyplot as plt

L = 1
N = 1000
dx = L/N
x = np.arange(N)*dx
sample_points = x[::10] # Sample every 10th point

f = lambda x: np.sin(2*np.pi*x)**4
# constant 1 function:
#f = lambda x: np.full_like(x, 1)
# linear function
#f = lambda x: x
#f = lambda x: np.sin(2*np.pi*x)

# example from exercise
# sin with periodic L
#f = lambda x: np.sin(2*np.pi*x)
# polynomial function f(x) = (x-x0)^2, with x0=L/2
#f = lambda x: (x-L/2)**2
# Step Function
#f = lambda x: 1 * (x > L/2)
# Sawtooth Function:
#f = lambda x: ((x*1000) % 500)/500


def fss_cd(xl, f, dx):
    return (f(xl + dx) - 2*f(xl) + f(xl - dx)) / dx**2

def fs_fw(xl, f, dx):
    print("xl=",xl," dx=",dx, " f(xl)=",f(xl)," f(xl + dx)=", f(xl+dx))
    return (f(xl + dx) - f(xl)) / dx




N_eval = 200
N_samples = 50

k2 = (np.arange(0,N_samples)-N_samples//2) * 2*np.pi
k2 = np.fft.fftshift(k2)
k2[N_samples//2] = 0


#first order derivative with fourier
fft = np.fft.fft(f(np.arange(N_samples)/N_samples))
ffts = 1j*k2*fft

fftpadded = np.zeros(N_eval, dtype='complex')

m = N_samples//2

fftpadded[0:m] = ffts[0:m]
fftpadded[N_eval-m:] = ffts[N_samples-m:]

# invers transformation (interpolation part)
fftinv_fourier_fs = np.fft.ifft(fftpadded, N_eval)*N_eval/N_samples



# first order derivative with fourier representation of finite differences
Dk_fourier_ip = (np.exp(1j*k2*dx)-1)/dx

ffts_fourier_ip = Dk_fourier_ip * fft

fftpadded_fourier_ip = np.zeros(N_eval, dtype='complex')

m = N_samples//2

fftpadded_fourier_ip[0:m] = ffts_fourier_ip[0:m]
fftpadded_fourier_ip[N_eval-m:] = ffts_fourier_ip[N_samples-m:]

fftinv_fourier_ip = np.fft.ifft(fftpadded_fourier_ip, N_eval)*N_eval/N_samples




# second order derivative with fourier
fftss = 1j*k2*ffts

fftpadded = np.zeros(N_eval, dtype='complex')

m = N_samples//2

fftpadded[0:m] = fftss[0:m]
fftpadded[N_eval-m:] = fftss[N_samples-m:]

# invers transformation (interpolation part)
fftinv = np.fft.ifft(fftpadded, N_eval)*N_eval/N_samples



dx_eval = L/N_eval
x_eval = np.arange(N_eval)*dx_eval


plt.plot(x, f(x), 'k-', label='actual function')
plt.plot(x_eval, fs_fw(x_eval, f, dx_eval), label='forward derivate')
plt.plot(np.arange(N_eval)/N_eval, fftinv_fourier_fs.real, 'b-', label='fft derivate')
plt.plot(np.arange(N_eval)/N_eval, fftinv_fourier_ip.real, 'r-', label='fft derivate finite differences')
plt.plot(x_eval, fss_cd(x_eval, f, dx), label='2nd order central difference scheme')
plt.plot(np.arange(N_eval)/N_eval, fftinv.real, 'b-', label='2nd order fft derivate')
plt.title('derivative');
plt.legend()
plt.show()