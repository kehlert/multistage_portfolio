#################################################
#################################################
## this file plots the efficient frontier
## the x-axis is the maximum allowed average value-at-risk (AV@R)
## the y-axis is the expected return given a maximum AV@R
## this file actually plots multiple frontiers, one for each set of correlated returns
#################################################
#################################################

import sys
import numpy as np
from markowitz import *
import math
import matplotlib.pyplot as plt

#####################
## set the model parameters
#####################
drift = {'S0': 0.03, 'S1': 0.035}
volatility = {'S0': 0.1, 'S1': 0.15}
interest_rate = 0.02
max_var = 10**4
capital = 10**4
time = 3
correlation = 0
# for i in range(0, 10):
#     expected_return, model, allocations, securities = get_markowitz_solution(capital,
#                                                                       drift,
#                                                                       volatility,
#                                                                       interest_rate,
#                                                                       max_var,
#                                                                       time,
#                                                                       correlation)
#     print allocations
#     for s in securities:
#         print('%s: %g' % (s,allocations[s]))
#     print '\n'

##########
## plot efficient frontier (x-axis = AV@R, y-axis = return as a %)
##########
#skip the correlation of 1, because it doesn't work with the code
correlations = np.linspace(-1, 1, num=9)[1:-1]
print correlations
max_std_devs = np.linspace(0, 5000, num=100)
max_return = 0

for corr in correlations:
    gurobi_percent_returns = []
    for std_dev in max_std_devs:
        gurobi, model, allocations, securities = get_markowitz_solution(capital,
                                                      drift,
                                                      volatility,
                                                      interest_rate,
                                                      std_dev**2,
                                                      time,
                                                      corr)
        gurobi_percent_returns.append(gurobi*100)
#         for s in allocations:
#             print('%s: %g' % (s,allocations[s].x))
#         print ''
    plt.plot(max_std_devs, gurobi_percent_returns, linestyle='-', label=r'$\rho=$%0.2f' % round(corr,2))
    max_return = max(max_return, max(gurobi_percent_returns))

plt.xlabel('Max Allowed Std. Deviation')
plt.ylabel('Return %')
plt.grid()
plt.ylim((5, math.ceil(max_return)))
plt.legend(loc=4)
plt.show()