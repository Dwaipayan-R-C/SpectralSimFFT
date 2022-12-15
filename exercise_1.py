import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,100,750)
y=x+3

x_samples = np.array([4,6,8,10])
y_samples = x_samples+3
print(x_samples)
print(y_samples)

plt.figure(1)
plt.plot(x,y)
plt.plot(x_samples,y_samples,'r*')

fft_x = np.fft.fft(x)
fft_x_sample = np.fft.fft(y_samples)

print(np.abs(fft_x_sample))
plt.figure(2)
plt.plot(np.real(fft_x_sample))
plt.plot(np.imag(fft_x_sample))

# plt.figure(3)
# plt.plot(np.abs(fft_x_sample))


plt.show()


