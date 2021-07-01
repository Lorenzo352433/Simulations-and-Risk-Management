"""
Binomial Option Pricing Model
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

def BinomialOptionPricingModel(S0, sigma, r, K, T, N):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    p_hat = (np.exp(r*dt)-d)/(u-d)
    price_tree = np.zeros([N+1,N+1])
    for i in range(N+1):
        for j in range(i+1):
            price_tree[j, i] = S0*(d**j) * (u**(i-j))
    option = np.zeros([N+1,N+1])
    option[:,N] = np.maximum(np.zeros(N+1), K - price_tree[:,N])
    for i in np.arange(N-1, -1, -1):
        for j in np.arange(0, i+1):
            option[j,i] = np.exp(-r*dt)*(p_hat*option[j, i+1]+(1-p_hat)*option[j+1, i+1])
    return option[0,0]

def BlackScholesEuropeanOptionPrice(S, K, rf, T, sigma):
    d1 = (np.log(S/(K)) + T*(rf + pow(sigma,2)/2.))/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return [S*norm.cdf(d1) - K*np.exp(-1*rf*T)*norm.cdf(d2), K*np.exp(-1*rf*T) * norm.cdf(-1*d2) - S*norm.cdf(-1*d1)]

print('Q1:{}'.format(BinomialOptionPricingModel(40, 0.2, 0.1, 32, 0.5, 50)))
print('Q2:{}'.format(BlackScholesEuropeanOptionPrice(40, 32, 0.1, 0.5, 0.2)[1]))

list_of_time_step_num = np.arange(5, 100, 5)
list_of_option_value = []

for single_time_step_num in list_of_time_step_num:
    list_of_option_value.append(BinomialOptionPricingModel(40, 0.2, 0.1, 32, 0.5, single_time_step_num))

fig, ax = plt.subplots()
ax.plot(list_of_time_step_num, list_of_option_value)
ax.set(xlabel='Number of Steps', ylabel='Option Value', title='Question 1')
ax.grid()
plt.show()

