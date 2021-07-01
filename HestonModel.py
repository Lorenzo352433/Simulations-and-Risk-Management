#Program Developed by Lorenzo based on Alan Cheung CQF 
#Contact me for more at lorenzoyatng@gmail.com or whatsapp +852 5112 2647

#Importing libraries
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
from scipy import stats
import math


#Black Scholes Try (Assume volatility stays constant)
def euroblackscholescall(S0,K,r,sigma,T):
	d1 = (math.log(S0/float(K)) + T*(r + pow(sigma,2)/2.))/float(sigma*math.sqrt(T))
	d2 = d1 - sigma * math.sqrt(T)
	return S0*norm.cdf(d1) - K*math.exp(-1*r*T)*norm.cdf(d2)

def euroblackscholesput(S0,K,r,sigma,T):
	d1 = (math.log(S0/float(K)) + T*(r + pow(sigma,2)/2.))/float(sigma*math.sqrt(T))
	d2 = d1 - sigma * math.sqrt(T)
	return norm.cdf(-d2)*K*math.exp(-1*r*T)-norm.cdf(-d1)*S0

print("With Black Scholes, the call price will be :")
print(euroblackscholescall(100,100,0.05,0.2,1))
print("With Black Scholes, the put price will be :")
print(euroblackscholesput(100,100,0.05,0.2,1))



#vt SDE
def montecarlovt(v0, xi, lamda, sigma, T, N, r):   	#I can't use lambda so i use lamda
	uniform_vector = np.random.uniform(size = N)
	normal_vector = stats.norm.ppf(uniform_vector)
	global brownian_vector1
	brownian_vector1 = normal_vector * np.sqrt(T)
	vt = v0+(lamda*(sigma**2-v0)*T)+(xi/np.sqrt(v0))*brownian_vector1*T
	vt = np.maximum(vt - xi,0)
	return np.mean(vt)
	print(brownian_vector1)

mcvt = montecarlovt(0.1,0.1,3,0.2,1,10000,0.05) #Start with v0=0.1, xi = 0.1, lambda = 2

print("The new volatility for the european option pricing will be:")
print(montecarlovt(0.1,0.1,3,0.2,1,10000,0.05))

def montecarloeurocall(S0, K, r, sigma, T, N,rho):
	uniform_vector = np.random.uniform(size = N)
	normal_vector = stats.norm.ppf(uniform_vector)
	brownian_vector = normal_vector * np.sqrt(T)
	St = S0+(r*S0*T)+S0*np.sqrt(mcvt)*(rho*brownian_vector1+np.sqrt(1-rho**2)*brownian_vector)*np.sqrt(T)
	payoff_vector = np.maximum(St - K,0)
	options_price_vector = np.exp(-1 * r * T) * payoff_vector
	call_option = np.cumsum(options_price_vector)/(1+np.arange(N))
	plt.title('European Call Option #With v0=0.1, xi = 0.1, lambda = 3')
	plt.xlabel('# Iterations')
	plt.ylabel('Estimated Value')
	plt.plot(call_option)
	plt.plot(np.full(N, euroblackscholescall(100,100,0.05,0.2,1)))
	plt.savefig("EurocallTry10.png")
	plt.show()
	print('The call option price with Heston Model will be:')
	print(call_option[9999])

mchscall =  montecarloeurocall(100, 100, 0.05, 0.2, 1, 10000,0.1)

def montecarloeuroput(S0, K, r, sigma, T, N,rho):
	uniform_vector = np.random.uniform(size = N)
	normal_vector = stats.norm.ppf(uniform_vector)
	brownian_vector = normal_vector * np.sqrt(T)
	St = S0+(r*S0*T)+S0*np.sqrt(mcvt)*(rho*brownian_vector1+np.sqrt(1-rho**2)*brownian_vector)*np.sqrt(T)
	payoff_vector = np.maximum(K-St,0)
	options_price_vector = np.exp(-1 * r * T) * payoff_vector
	put_option = np.cumsum(options_price_vector)/(1+np.arange(N))
	plt.title('European Put Option #With v0=0.1, xi = 0.1, lambda = 3')
	plt.xlabel('# Iterations')
	plt.ylabel('Estimated Value')
	plt.plot(put_option)
	plt.plot(np.full(N, euroblackscholesput(100,100,0.05,0.2,1)))
	plt.savefig('EuroputTry10.png')
	plt.show()
	print('The put option price with Heston Model will be:')
	print(put_option[9999])

mchsput =  montecarloeuroput(100, 100, 0.05, 0.2, 1, 10000,0.1)
