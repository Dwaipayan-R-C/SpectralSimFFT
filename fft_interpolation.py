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

h = (n-1)/2
h = n - 3
f_fwd_pad[0:h+1]   = f_fwd[0:h+1]
f_fwd_pad[npad-h:] = f_fwd[n-h:]

f_interpolated = np.fft.ifft(f_fwd_pad)*npad/n


plt.plot(x,f,'b*')
plt.plot(xpad,np.real(f_interpolated),color='r')
plt.legend(['Discrete samples','Interpolated curve'])



plt.show()

