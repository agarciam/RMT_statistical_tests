#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:45:23 2022

@author: andres.garcia.medina@uabc.edu.mx
"""


### librerias
import numpy as np
from scipy.linalg import eigh
from scipy import stats
import pandas as pd
import os


#######################################
path = os.getcwd()
significancias_onatski = pd.read_csv(path+'/data/tabla_onatski.csv',sep=',',\
                                     dtype=str,encoding='latin1',index_col=0)
    
significancias_tw = pd.read_csv(path+'/data/tabla_tracy_widom.csv',sep=' ',\
                                     header=None,dtype=str,encoding='latin1')


#######################################
### Algunas funciones utiles para limpieza de datos
def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''    
    for elem in listOfElems:
        if listOfElems.count(elem) > 1:
            return elem
    return False

class my_dictionary(dict): 

    def __init__(self): 
        self = dict() 
          
    def add(self, key, value): 
        self[key] = value 


def preproceso(data):
    data = data-data.mean(axis=0)
    data = data/data.std(axis=0)
    return(data)

def cov2corr(S):
    p = len(S)
    C = np.zeros([p,p])
    for i in range(p):
        for j in range(i,p):
            C[i,j] = S[i,j]/(np.sqrt(S[i,i])*np.sqrt(S[j,j]))
            C[j,i] = C[i,j]
            
    return(C)


#### Data imputation


def imputation(close, ratio=1):
    df_dataset = close
    if isinstance(close, pd.DataFrame)==True:
        n,p = df_dataset.shape
        markets = df_dataset.columns
        percentage = ratio
        dataset_imputed = my_dictionary() 
        for i in range(p):
            market = markets[i]
            if df_dataset[market].max() > 50000:
                 continue
            missing = df_dataset[market].isnull().sum()
            if  missing <= percentage*n/100:
                dataset_imputed.add(market, df_dataset[market])              
    
        prices_dataset_imputed = pd.DataFrame(dataset_imputed)  
        prices_dataset_imputed = prices_dataset_imputed.set_index(np.linspace(0,n-1,n))     
        prices_dataset_imputed = prices_dataset_imputed.astype(float).interpolate(method='linear', limit_direction='forward', order=3)
        prices_dataset_imputed = prices_dataset_imputed.set_index(df_dataset.index)
        returns = np.log(prices_dataset_imputed).diff().iloc[1:]
        returns_imputed = returns.set_index(np.linspace(0,n-2,n-1))  
        returns_imputed = returns_imputed.interpolate(method='linear', limit_direction='backward', order=3)
        returns_imputed = returns_imputed.set_index(returns.index)
        
    
    else:
        prices_dataset_imputed = df_dataset
        prices_dataset_imputed = prices_dataset_imputed.astype(float).interpolate(method='linear', limit_direction='forward', order=3)
        returns = np.log(prices_dataset_imputed).diff().iloc[1:]
        returns_imputed = returns.interpolate(method='linear', limit_direcion='backward', order=3)
    return(prices_dataset_imputed,returns_imputed)

#######################################
### Random matrix theory


### Filtering techique
def clipping_B(sample, T):
    eigval, eigvec = eigh(sample) 
    p = len(sample)
    n = T
    q = p/n
    dmax = (1.0 + np.sqrt(q))**2
    
    #replace noise
    noise = [] 
    k=0
    while (eigval[k] < dmax):
        noise.append(eigval[k])
        k+=1
    if len(noise) > 0:
        eignoise = np.mean(noise)
    count=p-k

    d = np.zeros([p])
    for i in range(p):
        if (eigval[i] < dmax):
            d[i] = eignoise
        else:
            d[i] = eigval[i]

    #reconstruct correlation matrix
    D = np.diag(d)
    XI_clipping = cov2corr(eigvec@D@eigvec.T)
    
    return(count,XI_clipping) 


#######################################
### Statistical test of  Onatski(2009)
def onatski(data,k0):
    n,p = data.shape
    data1 = data.iloc[:int(n/2),:]
    data2 = data.iloc[int(n/2):,:]
    n,p = data1.shape
    X = data1.values + data2.values*1j
    WUE = pd.DataFrame((1/n)*X.conjugate().T@X) 
    eigval_WUE, eigvec_GUE = eigh(WUE)

    # R statistics
    k1=7
    k1_k0 = k1-k0+1
    EIGS = eigval_WUE[::-1]
    index=10
    EIGENVALUES = np.zeros([index])
    for i in range(index):
        # R is invariant under change of scale and center
        EIGENVALUES[i] = (EIGS[i] - 2)*n**(-2/3)

    R = np.zeros([k1_k0])
    ii=0
    for i in range(k0,k1+1):
        statistics = (EIGENVALUES[i] - EIGENVALUES[i+1])/(EIGENVALUES[i+1] - EIGENVALUES[i+2])
        if i==k0:
            R[ii] = statistics
        else:
            R[ii] = max(R[ii-1],statistics)  
        ii+=1
    return(max(R))

def test_onatski(TABLA, j=0, level = 1):
    ### available levels: 1,2,3,4,5,6,7,8,9,10,15
    logical = TABLA[str(j)].values > significancias_onatski.loc[level].astype(float).values
    k = 0
    for value in logical:
        if  value==False:
            number = k
            break
        k+=1
        if k==8:
            number = 8
    return(number)


###  Tracy-Widom probabilities
def twprob(twtable, x):
    l = np.shape(twtable)[0]
    i = 0
    while(i < l and x >= twtable[i,0]):
        i += 1
    if(i == l):
        return(0)
    elif(i == 0):
        return(1)
    else:
        return(twtable[i-1,1]+(twtable[i,1] - twtable[i-1,1])*(x - twtable[i-1,0]) / (twtable[i,0]-twtable[i-1,0]))


### Probability(lambda > lambda_1) (2nd order)
def TW_Wishart_order2(p,n,twtable,lambda1,alpha):
    mu = (np.sqrt(n-0.5) + np.sqrt(p-0.5))**2
    sigma = np.sqrt(mu)*(1.0/np.sqrt(n-0.5)+1.0/np.sqrt(p-0.5))**(1./3)
    x_lambda1 = (n*lambda1-mu)/sigma
    probabilidad = twprob(twtable, x_lambda1)
    return(probabilidad)

def test_tw(eigval, n, alpha = 0.01):
    twtable = significancias_tw.astype(float).values
    p = len(eigval)
    k = 0
    for i in range(p):
        lambda_1 = eigval[-(i+1)]
        estadistico = TW_Wishart_order2(p,n,twtable, lambda_1,alpha)
        if estadistico < alpha:
            k +=1
        else:
            break
    return(k)

#######################################
### Sharpe Model
def resto_Sharpe(returns,M):
  dic={}
  bs={}
  for col in returns.columns:
    x=returns[col]
    lr=stats.linregress(M,x)
    a=lr.intercept
    b=lr.slope
    S=x-a-b*M
    dic[col]=S
    bs[col]=b
  df=pd.DataFrame(data=dic)
  return preproceso(df),bs
    
    
    
