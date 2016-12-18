from gurobipy import *
import math
import numpy as np
from scipy.stats import lognorm
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
correlation = np.array[[1,0],[0,1]] #correlation matrix of S0 and S1

alpha = 0.95
max_avar = 1000
branch_factor = 20 #number of branches stemming from each node in the scenario tree
stage_times = [0, 2, 3, 4]
#(probability, new rate)
#interest_rate_scenarios = [(0, 0.001), (1, 0.01)] 

#####################
## set up some global variables for convenience, also construct the scenario tree
#####################
stocks = drift.keys()
securities = stocks + ['B']
print(securities)

n_stages = len(stage_times)-1 #-1 because we sell everything at the final time
n_nodes_last_stage = branch_factor ** (n_stages-1)
n_scen = branch_factor ** (n_stages)

nodes_in_stage = {0: [0]}
n_nodes = 1
parents = {}
for i in range(1, n_stages):
    nodes_in_stage[i] = [id for id in range(n_nodes,n_nodes+branch_factor**i)]
    temp = 0
    for node in nodes_in_stage[i-1]:      
        all_children = nodes_in_stage[i]
        for j in range(0, branch_factor):
            child = all_children[temp*branch_factor + j]
            parents[child] = node 
        temp += 1
    n_nodes += branch_factor**i

#####################
## create the Gurobi model
#####################
m = Model('market')
m.params.LogFile = ''
#m.params.logtoconsole=0
m.modelSense = GRB.MAXIMIZE

#####################
## create the allocation variables
## these denote the number of each security bought, which can be a fraction
#####################
#key = (node, security)
allocations = {}
for node in range(0, n_nodes):
    for s in securities:
        allocations[node,s] = m.addVar(name='alloc_%s_%s' % (node,s))
	
m.update()

#force it to be a single-stage problem, helps with checking answer
# for i in range(0, n_stages):
#     first_node = nodes_in_stage[i][0]
#     for node in nodes_in_stage[i]:
#         for s in securities:
# 	        m.addConstr(allocations[first_node,s] == allocations[node,s])

#####################
## generate random returns
#####################
#key = (node, security)
#val = array of return history for up to that node for that security
returns = {}

#key = node (only nodes in the last stage are keys)
#value = array of random variables, representing the random returns before we sell
future_returns = {}
returns[0,'B'] = []
#bond returns
for i in range(1, n_stages):
    for node in nodes_in_stage[i]:
        #rates_unzipped = zip(*interest_rate_scenarios)
        #random_rates = np.random.choice(rates_unzipped[1], size=n_scen, p=rates_unzipped[0])
        ret = np.exp(interest_rate * (stage_times[i]-stage_times[i-1]))
        returns[node,'B'] = returns[parents[node],'B'] + [ret]

for node in nodes_in_stage[n_stages-1]:
    ret = np.exp(interest_rate * (stage_times[-1]-stage_times[-2]))
    future_returns[node,'B'] = [ret]

#stock returns
for s in stocks:
    returns[0,s] = []
    for i in range(1, n_stages):
        dt = stage_times[i]-stage_times[i-1]
        mean = drift[s] * dt - 0.5 * volatility[s]**2 * dt
        std = volatility[s] * math.sqrt(dt)
        rvs = np.random.lognormal(mean, std, len(nodes_in_stage[i]))
        index = 0
        for node in nodes_in_stage[i]:
            returns[node,s] = returns[parents[node],s] + [rvs[index]]
            index += 1

for s in stocks:    
    dt = stage_times[-1]-stage_times[-2]
    mean = drift[s] * dt - 0.5 * volatility[s]**2 * dt
    std = volatility[s] * math.sqrt(dt)
    for node in nodes_in_stage[n_stages-1]:
        future_returns[node,s] = np.random.lognormal(mean, std, branch_factor)

#####################
## setup the objective function for SAA
#####################
#set objective coefficients for variables in 2nd-to-last stage
#2nd-to-last, because we just sell everything in the final stage
for s in securities:
    for node in nodes_in_stage[n_stages-1]:
        ret = np.prod(returns[node,s])
        ret *= np.mean(future_returns[node,s])
        allocations[node,s].Obj = 1.0/n_nodes_last_stage * initial_prices[s] * ret

#####################
## setup the allocation constraints (assume we can't inject more capital)
#####################
#1st stage allocation constraint
m.addConstr(quicksum(allocations[0,s] * initial_prices[s] for s in securities) <= capital)

#allocation constraints for the other stages
for node in range(1, n_nodes):
    excess_stock_capital = 0
    for s in stocks:
        stock_alloc_diff = allocations[node,s] - allocations[parents[node],s]
        new_stock_price = initial_prices[s] * np.prod(returns[node,s])
        excess_stock_capital += new_stock_price * stock_alloc_diff
    bond_alloc_diff = allocations[node,'B'] - allocations[parents[node],'B']
    excess_bond_capital = initial_prices['B'] * np.prod(returns[node,'B']) * bond_alloc_diff
    m.addConstr(excess_stock_capital + excess_bond_capital == 0)

#####################
## Average value-at-risk constraint
## AV@R <= max_avar, where max_avar is specified at the top of the file
## also specified 'alpha' at the top of the file
#####################
gamma = m.addVar(lb=-1*GRB.INFINITY, name='gamma')
w = m.addVars(n_scen)
m.update()

w_sum = quicksum(w[k] for k in range(0, n_scen))
avar_constraint = m.addConstr(gamma + 1/(1-alpha) * 1.0/n_scen * w_sum <= max_avar)

w_index = 0
for node in nodes_in_stage[n_stages-1]:
    final_bond_capital = initial_prices['B'] * np.prod(returns[node,'B']) * future_returns[node,'B'][0] * allocations[node,'B']
    for i in range(0, branch_factor):
        final_stock_capital = quicksum(initial_prices[s] * np.prod(returns[node,s]) * future_returns[node,s][i] * allocations[node,s] for s in stocks)
        z_k = final_stock_capital + final_bond_capital
        m.addConstr(w[w_index] >= capital - z_k - gamma)
        w_index += 1

#####################
## optimize the model and report output
#####################
m.update()
m.optimize()
print('AV@R: %g' % (gamma + 1/(1-alpha) * 1.0/n_scen * w_sum).getValue())

print('Allocations:')
for s in sorted(securities):
	print('%s: %g' % (s, allocations[0,s].x * initial_prices[s] / capital))
	
# print('Average allocations for 2nd stage:')
# bond_allocs = []
# total_alloc = 0
# for i in range(0, branch_factor):
# 	bond_allocs.append(allocations['B',1,i].x)
# print('B: %g' % (np.mean(bond_allocs) * initial_prices[s]* returns['B',0]))
# total_alloc = np.mean(bond_allocs) * initial_prices[s]* returns['B',0]
# for s in sorted(stocks):
# 	stock_allocs = []
# 	for i in range(0, branch_factor):
# 		stock_allocs.append(allocations[s,1,i].x)
# 	total_alloc += np.mean(stock_allocs) * initial_prices[s] * returns[s,0][i]
# 	print('%s: %g' % (s, np.mean(stock_allocs) * initial_prices[s] * returns[s,0][i]))
# print(total_alloc)

#gives the percent return
percent_return = (m.objVal / capital - 1)*100
print('Projected Return: %0.2f%%' % round(percent_return,2))
sys.exit(0)
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
for node in nodes_in_stage[n_stages-1]:
    for i in range(0, branch_factor):
        final_stock_capital = quicksum(initial_prices[s] * np.prod(returns[node,s]) * future_returns[node,s][i] * allocations[node,s] for s in stocks)
        final_bond_capital = initial_prices['B'] * np.prod(returns[node,'B']) * future_returns[node,'B'][0] * allocations[node,'B']
        z_k = (final_stock_capital + final_bond_capital).getValue()
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

plt.grid()
plt.show()
