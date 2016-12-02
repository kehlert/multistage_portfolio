import math
import numpy as np
import matplotlib.pyplot as plt

#mean = 50 * exp(0.1 * 3) = 67.49294
#var = 580.786

x0 = 50
T = 3
mu = 0.1
sigma = 0.2
time_step = 0.01
n_steps = int(T / time_step)


n_runs = 10**2
n_runs_to_plot = 5
delta_bm = np.random.normal(0, math.sqrt(time_step), (n_steps, n_runs))
geometric_bm = np.zeros((n_steps+1, n_runs))
geometric_bm[0,:] = x0

for i in np.arange(0, n_steps):
	geometric_bm[i+1,:] = geometric_bm[i,:] * (1 + mu * time_step + sigma * delta_bm[i])

mean = np.mean(geometric_bm[n_steps-1,:])
var = np.var(geometric_bm[n_steps-1,:])
print('mean: %s' % mean)
print('var: %s' % var)
plt.plot(np.linspace(0, T, n_steps+1), geometric_bm[:,0:(n_runs_to_plot-1)])
plt.axis([0, T, 0, geometric_bm[:,0:(n_runs_to_plot-1)].max()+5])
plt.ylabel('X_t')
plt.xlabel('Time')
plt.show()
