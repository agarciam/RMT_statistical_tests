#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:44:32 2022

@author: andres.garcia.medina@uabc.edu.mx
"""

#######################################
# libraries

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd
import os
from functions_rmt_statistical_tests import checkIfDuplicates,preproceso,imputation,\
    clipping_B,test_tw, onatski, test_onatski, resto_Sharpe
import time
import json
import statsmodels.api as sm
import seaborn as sns

#######################################
### Parameters
start_time = time.time()
path = os.getcwd()
market = 'nasdaq'
case = 'high'
percentage = 5
start_date = '2005-01-03'
end_date = '2022-12-30'
#######################################
### Data and preprocessing

# Select market: S&P500 or NASDAQ
if market == 'sp500':
    index = 'S&P 500'
    close = pd.read_pickle(path+'/data/sp500_2022.pkl')
    MARKET = pd.read_csv(path+'/data/SPY.csv')
    MARKET['Date'] = MARKET["Date"].apply(lambda x: pd.Series(str(x).split(" "))).iloc[:,0]
    MARKET = MARKET.set_index(['Date'])
elif market == 'nasdaq':
    index = 'NASDAQ COMPOSITE'
    close = pd.read_pickle(path+'/data/nasdaq_2022.pkl')
    close.index = pd.to_datetime(close.index).strftime('%Y-%m-%d')
    MARKET = pd.read_csv(path+'/data/IXIC.csv', sep=',')
    MARKET = MARKET.set_index(['Date'])



# Period of time
close_window = close.loc[start_date:]
close_market = MARKET['Close'].loc[start_date:]


# Imputation
prices_market, returns_market = imputation(close_market)
prices_market = prices_market.loc[:end_date]
returns_market = returns_market.loc[:end_date]
prices, returns = imputation(close_window,ratio=percentage)
prices = prices.loc[:end_date]
returns = returns.loc[:end_date]


T, N = returns.shape
print(T,N, len(returns_market))


# Check for duplicate companies
tickers = returns.columns.to_list()
duplicados = checkIfDuplicates(tickers)
print(duplicados)



#######################################
# Ventanas moviles
print('moving window analysis:', market, case)
#######################################

if (case == 'high') and (market== 'sp500'):
    p_i = N
    q = 1/2
    m = 185
    step = 8

if (case == 'high') and (market== 'nasdaq'):
    p_i = N
    q = 1/2
    m = 114
    step = 8
    
elif (case == 'low') and (market== 'sp500'):
    p_i = N
    q = 1/8
    m = 60
    step = 4

elif (case == 'low') and (market== 'nasdaq'):
    p_i = 450
    q = 1/8
    m = 47
    step = 5


else:
    pass

n_i = int((1/q)*p_i)
delta = 20  
k1_k0 = 8

dates_in = []
number_signals_E = []
number_signals_tw_a1 = []
number_signals_tw_a5 = []
number_signals_tw_a10 = []
TABLA_S = pd.DataFrame(columns=['null'])
TABLA_R = pd.DataFrame(columns=['null'])
for i in range(m):

    
    print(i)
    
    start = i*delta
    end = i*delta + n_i

    R_window_in = preproceso(returns.iloc[start:end, :p_i])
    M=R_window_in.mean(axis=1)
    S,bs=resto_Sharpe(R_window_in, M)
    if market=='sp500':
        dates_in.append(R_window_in.index[-1].strftime("%Y-%m-%d"))
    else:
        dates_in.append(R_window_in.index[-1])        


    # Tracy-Widom
    E = R_window_in.corr()
    count_E, XI_E = clipping_B(E, n_i)  # Empirical
    number_signals_E.append(count_E)
    eigval_E, eigvec_E = eigh(E)
    number_signals_tw_a1.append(test_tw(eigval_E, n_i, alpha = 0.01))
    number_signals_tw_a5.append(test_tw(eigval_E, n_i, alpha = 0.05))
    number_signals_tw_a10.append(test_tw(eigval_E, n_i, alpha = 0.10))


    # Onatski
    R = np.zeros([k1_k0])
    for j in range(k1_k0):
        R[j] = onatski(R_window_in, j)
    R_statistics = pd.DataFrame(R, columns=[str(i)])
    TABLA_R = pd.concat([TABLA_R, R_statistics], axis=1)
    
    
    # Onatski without market
    R = np.zeros([k1_k0])
    for j in range(k1_k0):
      R[j] = onatski(S, j)
    R_statistics = pd.DataFrame(R, columns=[str(i)])
    TABLA_S = pd.concat([TABLA_S, R_statistics], axis=1)
TABLA_S = TABLA_S.drop(columns=['null'])    
TABLA_R = TABLA_R.drop(columns=['null'])



# Number of factors
NUMBER_FACTORS_ONATSKI_l1 = []
NUMBER_FACTORS_ONATSKI_l5 = []
NUMBER_FACTORS_ONATSKI_l10 = []
NUMBER_FACTORS_ONATSKI_S_l1 = []
NUMBER_FACTORS_ONATSKI_S_l5 = []
NUMBER_FACTORS_ONATSKI_S_l10 = []
for i in range(m):
    NUMBER_FACTORS_ONATSKI_l1.append(test_onatski(TABLA_R, j=i, level=1))
    NUMBER_FACTORS_ONATSKI_l5.append(test_onatski(TABLA_R,  j=i, level=5))
    NUMBER_FACTORS_ONATSKI_l10.append(test_onatski(TABLA_R,  j=i, level=10))
    NUMBER_FACTORS_ONATSKI_S_l1.append(test_onatski(TABLA_S, j=i, level=1))
    NUMBER_FACTORS_ONATSKI_S_l5.append(test_onatski(TABLA_S,  j=i, level=5))
    NUMBER_FACTORS_ONATSKI_S_l10.append(test_onatski(TABLA_S,  j=i, level=10))

number_factors = pd.DataFrame(list(zip(NUMBER_FACTORS_ONATSKI_l1, NUMBER_FACTORS_ONATSKI_l5,\
                                       NUMBER_FACTORS_ONATSKI_l10)),columns =['1%', '5%','10%'])
    
    
number_factors_S = pd.DataFrame(list(zip(NUMBER_FACTORS_ONATSKI_S_l1, NUMBER_FACTORS_ONATSKI_S_l5,\
                                       NUMBER_FACTORS_ONATSKI_S_l10)),columns =['1%', '5%','10%'])
    

number_components = pd.DataFrame(list(zip(number_signals_tw_a1, number_signals_tw_a5,\
                    number_signals_tw_a10,number_signals_E)),columns =['1%', '5%','10%','upper'])
#######################################
### plot and save figures

# number of factors
plt.figure()
x = np.linspace(0, m-1, m)
plt.axhline(y=k1_k0, color='r', linestyle='--')
plt.scatter(x, number_factors['1%'], marker="o",label='$\\alpha = 0.01$')
plt.scatter(x,number_factors['5%'], marker="^", label='$\\alpha = 0.05$')
plt.scatter(x,number_factors['10%'],marker="*", label='$\\alpha = 0.10$')
plt.xticks(np.arange(0, m, step), dates_in[::step])
plt.xticks(rotation=90)
plt.grid()
plt.legend()
plt.savefig(path+'/figures/Number_Factors_Onatski_'+market+'_'+case+'.png', dpi=600, bbox_inches='tight')
plt.show()

# number of factors without market
plt.figure()
x = np.linspace(0, m-1, m)
plt.axhline(y=k1_k0, color='r', linestyle='--')
plt.scatter(x, number_factors_S['1%'], marker="o",label='$\\alpha = 0.01$')
plt.scatter(x,number_factors_S['5%'], marker="^", label='$\\alpha = 0.05$')
plt.scatter(x,number_factors_S['10%'],marker="*", label='$\\alpha = 0.10$')
plt.xticks(np.arange(0, m, step), dates_in[::step])
plt.xticks(rotation=90)
plt.grid()
plt.legend()
plt.savefig(path+'/figures/Number_Factors_Onatski_S_'+market+'_'+case+'.png', dpi=600, bbox_inches='tight')
plt.show()


# number of components
plt.figure()
x = np.linspace(0, m-1, m)
plt.scatter(x,number_components['1%'], marker="o",label='$\\alpha = 0.01$')
plt.scatter(x,number_components['5%'], marker="^", label='$\\alpha = 0.05$')
plt.scatter(x,number_components['10%'],marker="*", label='$\\alpha = 0.10$')
plt.scatter(x, number_components['upper'], marker="s", label='$\lambda_{+}$')
plt.xticks(np.arange(0, m, step), dates_in[::step])
plt.xticks(rotation=90)
plt.grid()
plt.legend()
plt.savefig(path+'/figures/Number_Components_'+market+'_'+case+'.png', dpi=600, bbox_inches='tight')
plt.show()



#######################################
### save files
export_csv = TABLA_R.to_csv(path+'/files/Tabla_R_'+market+'_'+case+'.csv', header=True)
export_csv = number_factors.to_csv(path+'/files/number_factors_'+market+'_'+case+'.csv', header=True)
export_csv = TABLA_S.to_csv(path+'/files/Tabla_S_'+market+'_'+case+'.csv', header=True)
export_csv = number_factors_S.to_csv(path+'/files/number_factors_S_'+market+'_'+case+'.csv', header=True)
export_csv = number_components.to_csv(path+'/files/number_components_'+market+'_'+case+'.csv', header=True)

dates = list(dates_in)
if market=='sp500':
    dates.insert(0,returns.index[0].strftime("%Y-%m-%d"))
else:
    dates.insert(0,returns.index[0])
export_csv = pd.DataFrame(dates).to_csv(path+'/files/dates_range_'+market+'_'+case+'.csv', header=True)
export_csv = pd.DataFrame(R_window_in.columns.tolist()).to_csv(path+'/files/tickers_'+market+'_'+case+'.csv', header=True)




end_time = time.time()
elapsed = end_time - start_time
print(elapsed)



parameters = {'market': market, 'case':case,'elapsed':elapsed,\
              'percentage':percentage, 'start_date':start_date,'end_date':end_date,\
              'q':q, 'm':m, 'step':step, 'T':T,'N':N,'p_i':p_i,\
              'n_i':n_i,'delta':delta,'k1_k0':k1_k0}
out_file = open(path + '/files/parameters_'+str(market)+'_'+str(case)+'.json', "w") 
json.dump(parameters, out_file, indent = 6) 
out_file.close() 
