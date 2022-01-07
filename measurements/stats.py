import numpy as np
import pandas as pd

def auto_correlation(x,lag = 1):
	a = pd.Series(np.reshape(x,(-1)))
	b = a.autocorr(lag = lag)
	if np.isnan(b) or np.isinf(b):
		return 0
	return b

def acf(x,max_lag=1000):
	acf = []
	for i in range(max_lag):
		acf.append(auto_correlation(x,lag=i+1))
	return np.array(acf)

def Correlation(x,T=5, max_lag=25):
	corr=[]
	corr_diff=[]
	a = pd.Series(np.reshape(x, (-1)))
	Coarse = a.rolling(T).sum().abs()
	Fine = a.abs().rolling(T).sum()

	for i in range(-max_lag,max_lag+1,1):
		corr.append(Fine.corr(Coarse.shift(i)))
	for i in range(-max_lag,max_lag+1,1):
		corr_diff.append(Fine.corr(Coarse.shift(i))-Coarse.corr(Fine.shift(i)))
	return np.array(corr),np.array(corr_diff)

def Gain_or_loss_probability(x,theta=0.1):
	a = pd.Series(np.reshape(x, (-1)))
	b = [0] * 1001
	c = [0] * 1001
	for i in range(0,len(a)-1000,1):
		if i%100==0:
			print(i)
		for j in range(1,1001,1):
			if np.log(1+a[i:i+j].sum())>=theta:
				b[j]+=1
				break
		for j in range(1,1001,1):
			if np.log(1+a[i:i+j].sum())<=(-theta):
				c[j]+=1
				break
	b = b/np.sum(b)
	c = c/np.sum(c)
	return np.array(b),np.array(c)


