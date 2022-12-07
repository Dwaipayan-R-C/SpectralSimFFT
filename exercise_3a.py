import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

def fft_function(x,x_pad, flag=0):
    if(flag==0):
        f = np.sin(x) 
        f_pad = np.sin(x_pad)
    elif(flag==1):
        f=(x-len(x)/2)**2
        f_pad=(x_pad-len(x_pad)/2)**2
    elif(flag==2):
        f=signal.sawtooth(2 * np.pi * 5 * x)
        f_pad=signal.sawtooth(2 * np.pi * 5 * x_pad)
    return f,f_pad


n    = 10
npad = 300
duration = 2*np.pi
x    = np.arange(0., duration, duration/n)
xpad = np.arange(0., duration, duration/npad)

# Function definition
f,f_pad = fft_function(x,xpad,0)

# FFT transformation & interpolation
f_fwd = np.fft.fft(f)
f_fwd_pad = np.zeros(npad,dtype=complex)

cut_value = n - 6       # 7
f_fwd_pad[0:cut_value+1]   = f_fwd[0:cut_value+1]
f_fwd_pad[npad-cut_value:] = f_fwd[n-cut_value:]
f_interpolated = np.fft.ifft(f_fwd_pad)*npad/n

plt.figure(1)
plt.plot(x,f,'b*')
plt.plot(xpad,np.real(f_interpolated),color='r')
plt.legend(['Discrete samples','Interpolated curve'])
plt.title('Fourier interpolation on discrete spatial points')
plt.savefig('plots//exercise_3//curve_1_interpol.png')

plt.figure(2)
plt.plot(xpad, f_fwd_pad)

plt.figure(3)
plt.plot(x, f_fwd)

plt.show()