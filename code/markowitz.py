from gurobipy import *
import numpy as np
import math




#assumes we there are only two stocks
def get_markowitz_solution(capital,
                           drift,
                           volatility,
                           interest_rate, 
                           max_var,
                           time,
                           correlation):
    m = Model('markowitz')
    m.params.LogFile = ''
   
    drift_vec = np.array([drift[s] for s in sorted(drift.keys())])
    n_stocks = len(drift_vec)
    stocks = drift.keys()
    securities = stocks + ['B']
    cov = correlation * np.prod([v*math.sqrt(time) for v in volatility.values()])
    sigma = np.array([[volatility['S0'],cov],[cov,volatility['S1']]])
    print sigma
    sigma_inv = np.linalg.inv(sigma)
     
# the commented code uses Gurobi to optimize the Markowitz model
# since we have the Lagrange multiplier solution, this code isn't needed
# however, it's useful for checking the Lagrange multiplier solution

    alloc = {}
    stock_alloc = []
    #stocks
    for s in sorted(stocks):
        alloc[s] = m.addVar(obj = np.exp(drift[s]*time), name='alloc_%s' % s)
         #stock_alloc needs to be sorted
         #it gets multiplied by sigma, so the entries need to match up
        stock_alloc.append(alloc[s])
    stock_alloc = np.array(stock_alloc)

    #risk free asset
    alloc['B'] = m.addVar(obj = np.exp(interest_rate*time), name='allocation_r')

    m.modelSense = GRB.MAXIMIZE
    m.params.logtoconsole=0 
    m.update()

    a = np.transpose(stock_alloc).dot(sigma).dot(stock_alloc)

    m.addConstr(quicksum(alloc[a] for a in securities) <= capital)
    m.addQConstr(stock_alloc.dot(sigma).dot(stock_alloc) <= max_var)
    m.update()
    m.optimize()
    #print 'obj: %g' % m.objval
    #print 'var: %g' % stock_alloc.dot(sigma).dot(stock_alloc).getValue()
    return (m.objval/capital-1), m, alloc, securities
#     print('-----------\nGurobi Solution\n-----------')
#     print('Gurobi obj val: %g' % m.objVal)
#     var = stock_alloc.dot(sigma).dot(stock_alloc).getValue()
#     print('var of gurobi soln: %g' % var)
#     for a in securities:
#         print('%s: %g' % (a,alloc[a].x))
#     print('-------------------')
    
    ###############
    ## Calculate Lagrange multiplier solution
    ###############
    a = np.exp(drift_vec*time) - np.exp([interest_rate*time, interest_rate*time])
    denom = math.sqrt(a.dot(sigma_inv).dot(a))
    x = math.sqrt(max_var) * sigma_inv.dot(a) / denom
    xb = capital - (x[0] + x[1])

    expected_return = (xb * np.exp(interest_rate * time) + np.exp((drift_vec*time)).dot(x)) / capital - 1
    alloc = {'S0': x[0], 'S1': x[1], 'B': xb}
#     print('obj val: %g' % expected_return)
#     print('var: %g' % x.dot(sigma).dot(x))
#     print('S0: %g' % x[0])
#     print('S1: %g' % x[1])
#     print('B: %g' % xb)
    return gurobi, lagrange, m, alloc, securities