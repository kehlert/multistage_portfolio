from gurobipy import *
import math
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

np.random.seed(0)

n_scen = (10**2)

capital = 10000
interest_rate = 0.01
t = 3
alpha = 0.95
max_avar = 1000

initial_prices = {'B': 100, 'S0': 50, 'S1': 75}
drift = {'S0': 0.03, 'S1': 0.035}
volatility = {'S0': 0.1, 'S1': 0.15}
stocks = drift.keys()
securities = stocks + ['B']
print securities

m = Model('market')
m.params.LogFile = ''
m.params.logtoconsole=0
m.modelSense = GRB.MAXIMIZE

### initial allocation variables
allocations = {}
for s in securities:
	allocations[s] = m.addVar(name='alloc_%s_%s' % (s,1))
	
m.update()

### allocation constraint
m.addConstr(quicksum(allocations[s] * initial_prices[s] for s in securities) <= capital)

### generate random returns and setup the objective function for SAA
returns = {'B': math.exp(interest_rate * t)}
for s in stocks:
	mean = drift[s] * t - 0.5 * volatility[s]**2 * t
	std = volatility[s] * math.sqrt(t)
	#returns is of the form (1+percentage)
	#so (money we have) = return * (money we started with)
	returns[s] = np.random.lognormal(mean, std, n_scen)

for s in stocks:
    coeff = np.mean(returns[s])
    allocations[s].Obj = initial_prices[s] * coeff
		
allocations['B'].Obj = initial_prices['B'] * returns['B']
	
### AV@R <= max_avar constraint
gamma = m.addVar(lb=-1*GRB.INFINITY, name='gamma')
w = m.addVars(n_scen)
m.update()

print '\nAverage returns:'
for s in securities:
	avg_return = np.mean(returns[s]) - 1
	print '%s: %g' % (s,avg_return)

w_sum = quicksum(w[k] for k in range(0, n_scen))
avar_constraint = m.addConstr(gamma + 1/(1-alpha) * 1.0/n_scen * w_sum <= max_avar)

for k in range(0, n_scen):
	final_stock_capital = quicksum(allocations[s] * initial_prices[s] * returns[s][k] for s in stocks)
	final_bond_capital = allocations['B'] * initial_prices['B'] * returns['B']
	z_k = final_stock_capital + final_bond_capital
	m.addConstr(w[k] >= capital - z_k - gamma)

m.update()
m.optimize()
print 'avar: %g' % (gamma + 1/(1-alpha) * 1.0/n_scen * w_sum).getValue()

print('Allocations for AV@R <= %g:' % max_avar)
for s in sorted(securities):
	print '%s: %g' % (s, allocations[s].x * initial_prices[s] / capital)
	
end = quicksum(np.mean(allocations[s].x * initial_prices[s] * returns[s]) for s in stocks)
end += allocations['B'] * math.exp(t * interest_rate) * initial_prices['B']
print end.getValue() / capital - 1
print m.objVal
print('Projected Return: %g' % (m.objVal / capital - 1)) #subtract 1 so it's a percent return

##########
## test if that AV@R is correct
##########
losses = []
for k in range(0, n_scen):
    stock_returns = quicksum(initial_prices[s] * allocations[s] * returns[s][k] for s in stocks)
    bond_returns = initial_prices['B'] * returns['B'] * allocations['B']
    z_k = (stock_returns + bond_returns).getValue()
    loss = capital - z_k #positive if there's a loss
    losses.append(loss)
    
p = np.percentile(losses, alpha*100)
expected_gain = -1*np.mean(losses)
print ''
print('Expected gain: %g' % expected_gain)
print('Expected return: %g%%' % (expected_gain / capital * 100))
print('Monte Carlo V@R: %g' % p)
losses_exceeding_var = [l for l in losses if l > p]
avar = None
if losses_exceeding_var:
	avar = np.mean(losses_exceeding_var)
else:
	avar = float('nan')
print('Monte Carlo AV@R: %g' % avar)

##########
## plot efficient frontier (x-axis = AV@R, y-axis = return as a %)
##########
avars = np.linspace(0, 5000, num=101)
returns = []
for avar in avars:
	avar_constraint.RHS = avar
	m.update()
	m.optimize()
	returns.append((m.objVal / capital - 1)*100)
plt.plot(avars, returns, linestyle='-', marker='.')
plt.xlabel('Max Allowed Avg. Value at Risk ')
plt.ylabel('Return %')
plt.ylim(ymin=0, ymax=19)
plt.grid()
plt.show()
