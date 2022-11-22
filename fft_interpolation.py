import numpy as np
import matplotlib.pyplot as plt
L = 1
N = 1000
dx = L/N
x = np.arange(N)*dx
sample_points = x[::100] # Sample every 100th point
f = lambda x: np.sin(np.pi*x)**4
plt.plot(x, f(x), 'k-')
plt.plot(sample_points, f(sample_points), 'rx', ms=20) 
plt.title('A discretely sampled continuous function')
plt.show()
print()