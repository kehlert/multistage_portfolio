from gurobipy import *
import math
import numpy
from scipy.stats import lognorm
import matplotlib.pyplot as plt

#numpy.random.seed(0)

n_scen = int(math.pow(10, 4))

capital = 10000
interest_rate = 0.001
t1 = 2
t2 = 4
#(probability, new rate)
interest_rate_scenarios = [(0.3, 0.001), (0.7, 0.001)] 
alpha = 0.95
max_avar = 100

initial_price = {'S0': 100, 'S1': 100}
drift = {'S0': 0.03, 'S1': 0.035}
volatility = {'S0': 0.1, 'S1': 0.15}
stocks = initial_price.keys()
securities = stocks + ['B']
print securities

m = Model('market')
m.params.LogFile = ''
m.params.logtoconsole=0
m.modelSense = GRB.MAXIMIZE

### first stage vars
allocations = {}
for s in securities:
	allocations[s] = m.addVar(name='alloc_%s' % s)

m.update()

### allocation constraint
m.addConstr(quicksum(allocations[s] for s in securities) == 1)

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

allocations['B'].Obj = t1_returns['B'] * numpy.mean(t2_returns['B'])

for s in stocks:
	allocations[s].Obj = numpy.mean(t1_returns[s] * t2_returns[s])
	
### AV@R <= max_avar constraint
gamma = m.addVar(lb=-1*GRB.INFINITY, name='gamma')
w = m.addVars(n_scen)
m.update()

for s in stocks:
	print s, allocations[s].Obj
print 'B ' + str(allocations['B'].Obj)

w_sum = quicksum(w[k] for k in range(0, n_scen))
avar_constraint = m.addConstr(gamma + 1/(1-alpha) * 1.0/n_scen * w_sum <= max_avar)

for k in range(0, n_scen):
	stock_returns = quicksum(t1_returns[s][k] * t2_returns[s][k] * allocations[s] for s in stocks)
	bond_returns = t1_returns['B'] * t2_returns['B'][k] * allocations['B']
	z_k = capital * (stock_returns + bond_returns)
	m.addConstr(w[k] >= capital - z_k - gamma)

m.update()
m.optimize()

print('Allocations for AV@R <= %g:' % max_avar)
for s in sorted(securities):
	print '%s: %g' % (s, allocations[s].x)
	
print('Projected Return: %g' % (m.objVal - 1)) #subtract 1 so it's a percent return

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
# N = int(math.pow(10, 3))
# 
# new_returns = {'B': interest_rate}
# for s in stocks:
# 	mean = drift[s] * t - 0.5 * volatility[s]**2 * t
# 	var = volatility[s] * math.sqrt(t)
# 	new_returns[s] = numpy.random.lognormal(mean, var, N) - 1

# losses = []
# stock_bought = [s for s,alloc in allocations.items() if alloc.x > 0.9999]
# print stock_bought
# if len(stock_bought) > 1:
# 	raise Exception('bad allocation')
# stock_bought = stock_bought[0]
# 
# mean = drift[stock_bought] * t - 0.5 * volatility[stock_bought]**2 * t
# std = volatility[stock_bought] * math.sqrt(t)
# print 'mean-var'
# print mean
# print var
# quantile = lognorm.ppf(0.05, s=std, scale = math.exp(mean))
# print quantile
# value_at_risk = capital * (1-quantile)
# print 'value_at_risk: %g' % value_at_risk

# losses = []
# for k in range(0, N):
# 	stock_returns = quicksum((new_returns[s][k]+1) * allocations[s] for s in stocks)
# 	z_k = capital * (stock_returns + allocations['B'] * math.exp(interest_rate * t)).getValue()
# 	loss = capital - z_k #positive if there's a loss
# 	losses.append(loss)
# 
# p = numpy.percentile(losses, alpha*100)
# expected_gain = -1*numpy.mean(losses)
# print ''
# print('Expected gain: %g' % expected_gain)
# print('Expected return: %g%%' % (expected_gain / capital * 100))
# print('Monte Carlo V@R: %g' % p)
# print('Monte Carlo AV@R: %g' % numpy.mean([l for l in losses if l > p]))

##########
## plot efficient frontier (x-axis = AV@R, y-axis = return as a %)
##########

avars = numpy.linspace(0, 5000, num=101)
returns = []
for avar in avars:
	avar_constraint.RHS = avar
	m.update()
	m.optimize()
	returns.append((m.objVal - 1)*100)
plt.plot(avars, returns, linestyle='-', marker='.')
#plt.xticks(numpy.arange(min(avars), max(avars)+250, 250))
#plt.yticks(numpy.arange(0, max(returns)+0.01, 0.005))
plt.xlabel('Max Allowed Avg. Value at Risk ')
plt.ylabel('Return %')

plt.grid()
plt.show()
