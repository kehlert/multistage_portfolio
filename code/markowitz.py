from gurobipy import *
import numpy as np
import math

m = Model('Markowitz')
m.params.LogFile = ''

mus = {'S0': 0.1, 'S1': 0.12}
mus_vec = np.array([mus[s] for s in sorted(mus.keys())])
n_stocks = len(mus_vec)
stocks = mus.keys()
assets = stocks + ['r']
risk_free_rate = 0.02
max_var = 10**4
C = 10**4

sigma = np.array([[0.12,0.05],[0.05,0.1]])
print(sigma)
sigma_inv = np.linalg.inv(sigma)
print(sigma_inv)
print(np.linalg.eig(sigma))

alloc = {}
stock_alloc = []

#stocks
for s in stocks:
	alloc[s] = m.addVar(obj = mus[s], name='alloc_%s' % s)
	stock_alloc.append(alloc[s])
stock_alloc = np.array(stock_alloc)

#risk free asset
alloc['r'] = m.addVar(obj = risk_free_rate, name='allocation_r')

m.modelSense = GRB.MAXIMIZE
m.update()

#a = np.transpose(stock_alloc).dot(sigma).dot(stock_alloc)

m.addConstr(quicksum(alloc[a] for a in assets) <= C)
m.addQConstr(stock_alloc.dot(sigma).dot(stock_alloc) <= max_var)
m.optimize()
print('-----------\nGurobi Solution\n-----------')
print('Gurobi obj val: %g' % m.objVal)
var = stock_alloc.dot(sigma).dot(stock_alloc).getValue()
print('var of gurobi soln: %g' % var)
for a in assets:
	print('%s: %g' % (a,alloc[a].x))
print('-------------------')

###############
## Calculate Langrange multiplier solution
###############
a = mus_vec - np.array([risk_free_rate, risk_free_rate])
denom = math.sqrt(a.dot(sigma_inv).dot(a))
x = math.sqrt(max_var)/1.0 * sigma_inv.dot(a) / denom
xr = C - (x[0] + x[1])

obj = risk_free_rate * xr + mus_vec.dot(x)
print('obj val: %g' % obj)
print('var: %g' % x.dot(sigma).dot(x))
print('S0: %g' % x[0])
print('S1: %g' % x[1])
print('r: %g' % xr)