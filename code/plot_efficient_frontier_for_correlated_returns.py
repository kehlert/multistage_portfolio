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
from multi_stage import *
import math
import matplotlib.pyplot as plt

np.random.seed(0)

#####################
## set the model parameters
#####################
capital = 10000
interest_rate = 0.01
initial_prices = {'B': 100, 'S0': 50, 'S1': 75}
drift = {'S0': 0.03, 'S1': 0.035}
volatility = {'S0': 0.1, 'S1': 0.15}
correlation_mat = np.array([[1,0],[0,1]]) #correlation matrix of S0 and S1
cost = 2

alpha = 0.95
max_avar = 100
branch_factor = 60 #number of branches stemming from each node in the scenario tree
stage_times = [0, 1, 3]
#(probability, new rate)
#interest_rate_scenarios = [(0, 0.001), (1, 0.01)] 

#####################
## construct the model
#####################
# model, securities, allocations, avar_constraint = get_multi_stage_model(capital,
#                                                                         interest_rate,
#                                                                         initial_prices,
#                                                                         drift,
#                                                                         volatility,
#                                                                         correlation_mat,
#                                                                         alpha,
#                                                                         max_avar,
#                                                                         branch_factor,
#                                                                         stage_times,
#                                                                         cost)                

####################
## optimize and report output
####################
# model.optimize()
# print('Allocations:')
# for s in sorted(securities):
#     print('%s: %g' % (s, allocations[0,s].x * initial_prices[s] / capital))
# 
# #gives the percent return
# percent_return = (model.objVal / capital - 1)*100
# print('Projected Return: %0.2f%%' % round(percent_return,2))

#sys.exit(0)

##########
## plot efficient frontier (x-axis = AV@R, y-axis = return as a %)
##########
#skip the correlation of -1 and 1, because it doesn't work with the code
# correlations = np.linspace(-1, 1, num=9)[1:-1]
# print correlations
# avars = np.linspace(-300, 3000, num=16)
# max_return = 0
# N = 150
# returns = {}
# for corr in correlations:
#     returns[corr] = {}
#     for avar in avars:
#         returns[corr][avar] = []
# for i in range(0, N):
#     print('N: %g' % i)
#     for corr in correlations:
#         correlation_mat = np.array([[1,corr],[corr,1]])
#         model, securities, allocations, avar_constraint = get_multi_stage_model(capital,
#                                                                                 interest_rate,
#                                                                                 initial_prices,
#                                                                                 drift,
#                                                                                 volatility,
#                                                                                 correlation_mat,
#                                                                                 alpha,
#                                                                                 max_avar,
#                                                                                 branch_factor,
#                                                                                 stage_times,
#                                                                                 cost)
#         for avar in avars:
#             avar_constraint.RHS = avar
#             model.update()
#             model.optimize()
#             returns[corr][avar].append((model.objVal / capital - 1)*100)
# 
# for corr in correlations:
#     mean_return = []
#     for avar in avars:
#         mean_return.append(np.mean(returns[corr][avar]))
#     plt.plot(avars, mean_return, linestyle='-', marker='.', label=r'$\rho=$%0.2f' % round(corr,2))
#     max_return = max(max_return, max(mean_return))
# plt.xlabel('Max Allowed Avg. Value at Risk ')
# plt.ylabel('Return %')
# plt.grid()
# plt.ylim((0, math.ceil(max_return)+1))
# plt.legend(loc=4)
# plt.show()

##########
## plot efficient frontier, compare increasing cost (x-axis = AV@R, y-axis = return as a %)
##########
costs = np.linspace(0, 5, num=6)
avars = np.linspace(-300, 3000, num=16)
max_return = 0
N = 150
returns = {}
for cost in costs:
    returns[cost] = {}
    for avar in avars:
        returns[cost][avar] = []
for i in range(0, N):
    print('N: %g' % i)
    for cost in costs:
        correlation_mat = np.array([[1,0],[0,1]])
        model, securities, allocations, avar_constraint = get_multi_stage_model(capital,
                                                                                interest_rate,
                                                                                initial_prices,
                                                                                drift,
                                                                                volatility,
                                                                                correlation_mat,
                                                                                alpha,
                                                                                max_avar,
                                                                                branch_factor,
                                                                                stage_times,
                                                                                cost)
        for avar in avars:
            avar_constraint.RHS = avar
            model.update()
            model.optimize()
            returns[cost][avar].append((model.objVal / capital - 1)*100)

for cost in costs:
    mean_return = []
    for avar in avars:
        mean_return.append(np.mean(returns[cost][avar]))
    plt.plot(avars, mean_return, linestyle='-', marker='.', label=r'$\cost=$%0.2f' % cost)
    max_return = max(max_return, max(mean_return))
plt.xlabel('Max Allowed Avg. Value at Risk ')
plt.ylabel('Return %')
plt.grid()
plt.ylim((0, math.ceil(max_return)+1))
plt.legend(loc=4)
plt.show()