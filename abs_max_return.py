from gurobipy import *
import math
import numpy
from scipy.stats import lognorm
import matplotlib.pyplot as plt

numpy.random.seed(0)

n_scen = int(math.pow(10, 4))

capital = 10000
interest_rate = 0.001
t1 = 2
t2 = 4
#(probability, new rate)
interest_rate_scenarios = [(0, 0.001), (1, 0.01)] 
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

### first and second stage vars
allocations_1 = {}
allocations_2 = {}
for s in securities:
	allocations_1[s] = m.addVar(name='alloc_%s_%s' % (s,1))
	for k in range(0, n_scen):
		allocations_2[s,k] = m.addVar(name='alloc_%s_%s' % (s,k))
	
m.update()

### 1st stage allocation constraint
m.addConstr(quicksum(allocations_1[s] * initial_prices[s] for s in securities) <= capital)

#TODO remove
for s in securities:
	for k in range(0, n_scen):
		pass
		#m.addConstr(allocations_2[s,k] == allocations_1[s])

### generate random returns and setup the objective function for SAA
t1_returns = {'B': math.exp(interest_rate * t1)}

rates_unzipped = zip(*interest_rate_scenarios)
random_rates = numpy.random.choice(rates_unzipped[1], size=n_scen, p=rates_unzipped[0])
t2_returns = {'B': numpy.exp(random_rates * (t2-t1))}

for s in stocks:
	mean1 = drift[s] * t1 - 0.5 * volatility[s]**2 * t1
	std1 = volatility[s] * math.sqrt(t1)
	t1_returns[s] = numpy.random.lognormal(mean1, std1, n_scen)
	
 	mean2 = drift[s] * (t2-t1) - 0.5 * volatility[s]**2 * (t2-t1)
 	std2 = volatility[s] * math.sqrt((t2-t1))
 	t2_returns[s] = numpy.random.lognormal(mean2, std2, n_scen)

for s in stocks:
	for k in range(0, n_scen):
		allocations_2[s,k].Obj = 1.0/n_scen * initial_prices[s] * t1_returns[s][k] * t2_returns[s][k]
		
for k in range(0, n_scen):
	allocations_2['B',k].Obj = 1.0/n_scen * initial_prices['B'] * t1_returns['B'] * t2_returns['B'][k]
 	
### 2nd stage allocation constraint
### need to set this up after generating returns
for k in range(0, n_scen):
	excess_stock_capital = 0
	for s in stocks:
		stock_alloc_diff = allocations_1[s] - allocations_2[s,k]
		new_stock_price = initial_prices[s] * t1_returns[s][k]
		excess_stock_capital += new_stock_price * stock_alloc_diff
	bond_alloc_diff = allocations_1['B'] - allocations_2['B',k]
	excess_bond_capital = initial_prices['B'] * math.exp(interest_rate * t1) * bond_alloc_diff
	m.addConstr(excess_stock_capital + excess_bond_capital == 0)
	
### AV@R <= max_avar constraint
gamma = m.addVar(lb=-1*GRB.INFINITY, name='gamma')
w = m.addVars(n_scen)
m.update()

print '\nAverage returns:'
for s in securities:
	avg_return = numpy.mean(t1_returns[s] * t2_returns[s]) - 1
	print '%s: %g' % (s,avg_return)

w_sum = quicksum(w[k] for k in range(0, n_scen))
avar_constraint = m.addConstr(gamma + 1/(1-alpha) * 1.0/n_scen * w_sum <= max_avar)

for k in range(0, n_scen):
	final_stock_capital = quicksum(initial_prices[s] * t1_returns[s][k] * t2_returns[s][k] * allocations_2[s,k] for s in stocks)
	final_bond_capital = initial_prices['B'] * t1_returns['B'] * t2_returns['B'][k] * allocations_2['B',k]
	z_k = final_stock_capital + final_bond_capital
	m.addConstr(w[k] >= capital - z_k - gamma)

m.update()
m.optimize()
print 'avar: %g' % (gamma + 1/(1-alpha) * 1.0/n_scen * w_sum).getValue()

print('Allocations for AV@R <= %g:' % max_avar)
for s in sorted(securities):
	print '%s: %g' % (s, allocations_1[s].x * initial_prices[s] / capital)
	
print('Average allocations for 2nd stage:')
bond_allocs = []
total_alloc = 0
for k in range(0, n_scen):
	bond_allocs.append(allocations_2['B',k].x)
print 'B: %g' % (numpy.mean(bond_allocs) * initial_prices[s]* t1_returns['B'])
total_alloc = numpy.mean(bond_allocs) * initial_prices[s]* t1_returns['B']
for s in sorted(stocks):
	stock_allocs = []
	for k in range(0, n_scen):
		stock_allocs.append(allocations_2[s,k].x)
	total_alloc += numpy.mean(stock_allocs) * initial_prices[s] * t1_returns[s][k]
	print '%s: %g' % (s, numpy.mean(stock_allocs) * initial_prices[s] * t1_returns[s][k])
print total_alloc
	
end = quicksum(numpy.mean(allocations_1[s].x * initial_prices[s] * t1_returns[s]) for s in stocks)
end += allocations_1['B'] * math.exp(t1 * interest_rate) * initial_prices['B']
print end.getValue() / capital - 1
print('Projected Return: %g' % (m.objVal / capital - 1)) #subtract 1 so it's a percent return

##########
## calculate AV@R of above solution
##########

# avar_model = Model('avar')
# avar_model.params.logtoconsole=0
# 
# gamma = avar_model.addVar(lb=-1*GRB.INFINITY, obj=1, name='gamma')
# w = {}
# for k in range(0, n_scen):
# 	w[k] = avar_model.addVar(obj=1/(1-alpha) * 1.0/n_scen, name='w_%g' % k)
# 
# avar_model.update()
# 
# for k in range(0, n_scen):
# 	stock_returns = quicksum((returns[s][k]+1) * allocations[s] for s in stocks)
# 	z_k = capital * (stock_returns + allocations['B'] * math.exp(interest_rate * t)).getValue()
# 	avar_model.addConstr(w[k] + gamma >= capital - z_k)
# 	
# avar_model.update()
# avar_model.optimize()

##########
## test if that AV@R is correct
##########
losses = []
for k in range(0, n_scen):
	stock_returns = quicksum(initial_prices[s] * t1_returns[s][k] * t2_returns[s][k] * allocations_2[s,k] for s in stocks)
	bond_returns = initial_prices['B'] * t1_returns['B'] * t2_returns['B'][k] * allocations_2['B',k]
	z_k = (stock_returns + bond_returns).getValue()
	loss = capital - z_k #positive if there's a loss
	losses.append(loss)

print max(losses)
p = numpy.percentile(losses, alpha*100)
expected_gain = -1*numpy.mean(losses)
print ''
print('Expected gain: %g' % expected_gain)
print('Expected return: %g%%' % (expected_gain / capital * 100))
print('Monte Carlo V@R: %g' % p)
losses_exceeding_var = [l for l in losses if l > p]
avar = None
if losses_exceeding_var:
	avar = numpy.mean(losses_exceeding_var)
else:
	avar = float('nan')
print('Monte Carlo AV@R: %g' % avar)

##########
## plot efficient frontier (x-axis = AV@R, y-axis = return as a %)
##########
avars = numpy.linspace(0, 3000, num=101)
returns = []
for avar in avars:
	avar_constraint.RHS = avar
	m.update()
	m.optimize()
	returns.append((m.objVal / capital - 1)*100)
plt.plot(avars, returns, linestyle='-', marker='.')
#plt.xticks(numpy.arange(min(avars), max(avars)+250, 250))
#plt.yticks(numpy.arange(0, max(returns)+0.01, 0.005))
plt.xlabel('Max Allowed Avg. Value at Risk ')
plt.ylabel('Return %')

plt.grid()
plt.show()
