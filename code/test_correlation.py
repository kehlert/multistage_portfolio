import numpy as np
import sys
drift = {'S0': 0.03, 'S1': 0.035, 'S2': 0.025}
volatility = {'S0': 0.1, 'S1': 0.15, 'S2': 0.2}
stocks = drift.keys()

W = np.random.rand(len(stocks),len(stocks))
diag = np.diag(np.random.rand(len(stocks))) + np.diag([1]*len(stocks))
#this matrix is guaranteed to be pos. def by Gershgorin's circle theorem
A = np.transpose(W).dot(W) + diag
diag_A = np.diag(A)
B = np.diag(1.0/np.sqrt(diag_A))
correlation = B.dot(A).dot(B)
print('expected correlation: ')
print(correlation)

L = np.linalg.cholesky(correlation)
N = 10**6
normal_rvs = np.random.normal(size = (len(stocks),N))
correlated_normal_rvs = dict(zip(stocks, L.dot(normal_rvs)))

all_returns = []
for s in stocks:
    dt = 1
    mean = (drift[s] - 0.5 * volatility[s]**2) * dt
    ret = mean + volatility[s]*correlated_normal_rvs[s]
    rvs = np.exp(ret)
    if len(all_returns) != 0:
        all_returns = np.vstack((all_returns, ret))
    else:
        all_returns = ret
    print('%s mean: %g' % (s,np.mean(rvs)))
    print('%s var: %g' % (s,np.var(rvs)))
print('correlations:')
print(np.corrcoef(all_returns))