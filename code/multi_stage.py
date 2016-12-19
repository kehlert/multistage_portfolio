#################################################
#################################################
## this file sets up a Gurobi model
## the model is of a portfolio, where we can choose between a bond and some given stocks
## see "plot_efficient_frontier_for_correlated_returns.py" for an example of how to
## use this file
#################################################
#################################################

from gurobipy import *
import math
import numpy as np
from scipy.stats import lognorm

def get_multi_stage_model(capital, interest_rate, initial_prices, drift, volatility,
                          correlation_mat, alpha, max_avar, branch_factor,stage_times, cost):
    #####################
    ## set up some global variables for convenience, also construct the scenario tree
    #####################
    stocks = drift.keys()
    securities = stocks + ['B']
    print(securities)

    #used to create correlated returns
    L = np.linalg.cholesky(correlation_mat)

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
    m = Model('portfolio')
    m.params.LogFile = ''
    m.params.logtoconsole=0  
    m.modelSense = GRB.MAXIMIZE

    #####################
    ## create the allocation variables
    ## these denote the number of each security bought, which can be a fraction
    #####################
    #key = (node, security)
    allocations = {}
    number_traded = {}
    for node in range(0, n_nodes):
        for s in securities:
            allocations[node,s] = m.addVar(name='alloc_%s_%s' % (node,s))
    for node in range(1, n_nodes):
        for s in securities:
            number_traded[node,s] = m.addVar(obj = -1.0*cost / (n_nodes-1),
                                             lb=-1*GRB.INFINITY,
                                             name='number_traded_%s_%s' % (node, s))
      
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
        #key = stock name
        #value = correlated normal random variable
        normal_rvs = np.random.normal(size = (len(stocks),len(nodes_in_stage[i])))
        correlated_normal_rvs = dict(zip(stocks, L.dot(normal_rvs)))
        dt = stage_times[i]-stage_times[i-1]
        for s in stocks: 
            mean = (drift[s] - 0.5 * volatility[s]**2) * dt
            ret = mean + volatility[s] * correlated_normal_rvs[s]
            #rvs = np.random.lognormal(mean, std, len(nodes_in_stage[i]))
            log_norm_rvs = np.exp(ret)
            index = 0
            for node in nodes_in_stage[i]:
                returns[node,s] = returns[parents[node],s] + [log_norm_rvs[index]]
                index += 1

    dt = stage_times[-1]-stage_times[-2]
    for node in nodes_in_stage[n_stages-1]:
        normal_rvs = np.random.normal(size = (len(stocks),branch_factor))
        #key = stock name
        #value = correlated normal random variable
        correlated_normal_rvs = dict(zip(stocks, L.dot(normal_rvs)))
        for s in stocks:
            mean = drift[s] * dt - 0.5 * volatility[s]**2 * dt
            ret = mean + volatility[s] * correlated_normal_rvs[s]
            future_returns[node,s] = np.exp(ret)

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
            #this makes number_traded == |difference|
            m.addConstr(number_traded[node,s] >= allocations[node,s] - allocations[parents[node],s])
            m.addConstr(number_traded[node,s] >= allocations[parents[node],s] - allocations[node,s])
            stock_alloc_diff = allocations[node,s] - allocations[parents[node],s]
            new_stock_price = initial_prices[s] * np.prod(returns[node,s])
            excess_stock_capital += new_stock_price * stock_alloc_diff
        #this makes number_traded == |difference|
        m.addConstr(number_traded[node,'B'] >= allocations[node,'B'] - allocations[parents[node],'B'])
        m.addConstr(number_traded[node,'B'] >= allocations[parents[node],'B'] - allocations[node,'B'])
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

    m.update()        
    return m, securities, allocations, avar_constraint
    
    
##########
## THIS IS JUST EXTRA CODE, not sure if I should get rid of it yet
## test if that AV@R is correct by just looking at the (1-alpha) percentile of the losses
##########
# print('AV@R: %g' % (gamma + 1/(1-alpha) * 1.0/n_scen * w_sum).getValue())
#
# losses = []
# for node in nodes_in_stage[n_stages-1]:
#     for i in range(0, branch_factor):
#         final_stock_capital = quicksum(initial_prices[s] * np.prod(returns[node,s]) * future_returns[node,s][i] * allocations[node,s] for s in stocks)
#         final_bond_capital = initial_prices['B'] * np.prod(returns[node,'B']) * future_returns[node,'B'][0] * allocations[node,'B']
#         z_k = (final_stock_capital + final_bond_capital).getValue()
#         loss = capital - z_k #positive if there's a loss
#         losses.append(loss)
# 
# p = np.percentile(losses, alpha*100)
# expected_gain = -1*np.mean(losses)
# print ''
# print('Expected gain: %g' % expected_gain)
# print('Expected return: %g%%' % (expected_gain / capital * 100))
# print('V@R based on losses percentile: %g' % p)
# losses_exceeding_var = [l for l in losses if l > p]
# avar = None
# if losses_exceeding_var:
# 	avar = np.mean(losses_exceeding_var)
# else:
# 	avar = float('nan')
# print('AV@R based on s losses percentile: %g' % avar)
