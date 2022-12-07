import numpy as np
import matplotlib.pyplot as plt

L=1
N=1000
dx = L/N
x = np.arange(N)*dx
sample_points = x[::10]

f:lambda x:np.sin(2*np.pi*x)**4

def finite_diff_1(xl,f,dx):
    return (f(xl + dx) - 2*f(xl) + f(xl - dx)) / dx**2

def fs_fw(xl, f, dx):
    print("xl=",xl," dx=",dx, " f(xl)=",f(xl)," f(xl + dx)=", f(xl+dx))
    return (f(xl + dx) - f(xl)) / dx

N_eval = 200
N_samples = 50

k2 = (np.arange(0,N_samples)-N_samples//2) * 2*np.pi
k2 = np.fft.fftshift(k2)
k2[N_samples//2] = 0