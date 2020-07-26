import numpy as np
import matplotlib.pyplt as plt

x = np.random.rand(100,1)
A = np.random.rand(100,100)

# random matrix multiplication
maxiters = 1000
u = x.copy()
for iters in range(maxiters):
    u = A*u
plt.figure()
plt.plot(u)

# random matrix multiplication with relu
maxiters = 1000
u = x.copy()
for iters in range(maxiters):
    u = A*u
    u = u()
plt.figure()
plt.plot(u)

maxiters = 1000
u = x.copy()
for iters in range(maxiters):
    u = A * u
plt.figure()
plt.plot(u)
