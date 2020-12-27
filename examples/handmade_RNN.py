import numpy as np
import matplotlib.pyplot as plt

stable_r1 = 1
stable_r2 = 1
input_r1 = 2
input_r2 = 3
b1 = 0.1
b2 = 2
alpha = 0.1
noise_sd = 0.1
ntimestep = 1000

r0 = np.array([[input_r1], [input_r2]])
b = np.array([[b1], [b2]])
w_rnn = np.array([[0, (stable_r1-b1)/stable_r2],
                 [(stable_r2-b2)/stable_r1, 0]])
r = np.zeros((ntimestep+1, 2, 1))
r[0, :, :] = r0
for t in range(ntimestep):
    r[t+1, :, :] = (1-alpha)*r[t, :, :] + alpha*((w_rnn @ r[t, :, :]) + b) + np.random.normal(0, noise_sd, (2, 1))

r = r[:, :, 0]

plt.clf()
plt.plot(r)
