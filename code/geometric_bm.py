import sys
import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

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
#plt.show()

################
## correlated GBM
################
T = 60
time_step = 0.1
n_steps = int(T / time_step)
dim = 5
n_runs = 10**5
x0 = np.array([50, 20, 30, 45, 70])
mus = np.array([0.1,0.05,0.12,0.2,0.08])
sigmas = np.array([0.2, 0.15, 0.3, 0.12, 0.15])

#upper = np.matrix([[0,0.5,0.5,0.5,0.5],[0,0,0.5,0.5,0.5],[0,0,0,0.5,0.5],[0,0,0,0,0.5],[0,0,0,0,0]])
W = np.random.rand(dim, dim)
#B is pos. def. by Gershgorin's theorem

#use for positive correlation
B = np.dot(W, np.transpose(W))

#use for negative correlation
#B = np.dot(W, np.transpose(W)) + 3*np.diag([2, 3, 3, 5, 4])

E = np.diag(np.sqrt(1./np.diagonal(B))) #used to put ones on the diagonals
corr = E.dot(B).dot(E)
print('\nCorrelation matrix: ')
print(corr)
L = np.linalg.cholesky(corr)

prices_at_final_time = np.empty((dim, n_runs))

delta_bm = np.random.normal(0, math.sqrt(time_step), (dim, n_runs, n_steps))

for d in np.arange(0, dim):
	corr_geometric_bm = np.zeros((n_steps+1, n_runs))
	corr_geometric_bm[0,:] = x0[d]

	for i in np.arange(0, n_steps):
		#for delta_corr_bm: rows = which time point, columns = which run
		delta_corr_bm = np.transpose(L[d,:].dot(delta_bm[:,:,i]))
		corr_geometric_bm[i+1,:] = corr_geometric_bm[i,:] * (1 + mus[d] * time_step + sigmas[d] * delta_corr_bm)
	
	prices_at_final_time[d,:] = corr_geometric_bm[-1,:]

print('\nExpected means: ')
expected_means = x0 * np.exp(mus * T)
print(expected_means)

print('\nSample means: ')
sample_means = np.mean(prices_at_final_time, axis = 1)
print(sample_means)

print('\nSample correlation coefficients:')
corr_coeffs = np.corrcoef(prices_at_final_time, rowvar = 1)
print(corr_coeffs)
		
plt.plot(np.linspace(0, T, n_steps+1), corr_geometric_bm[:,0:(n_runs_to_plot-1)])
plt.axis([0, T, 0, corr_geometric_bm[:,:].max()])
plt.ylabel('X_t')
plt.xlabel('Time')
#plt.show()