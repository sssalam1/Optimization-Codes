"""
@author: Salam Saudagar
Multiobjective Optimization using Genetic Algorithm
"""
import numpy as np
import pandas as pd

def fun1(x):
    return(np.power(x,2))

def fun2(x):
    return(np.power( (x-2) , 2) )
    
nBits = 6
XL,XU = -1000, 1000
popl = 50 ;Pc = 0.7 ; Pm = 1/popl

####==== Function to Create the populations
def CreatePopulation(popl, nBits):
    popl_binary = []
    for i in range(popl):
        binr = np.random.choice([0, 1], size=nBits)
        popl_binary.append(binr)
    return popl_binary

####==== Function to Convert the binary number to decimal
def bin2dec(Init_popl, nBits, popl):
    X=[]
    for j in range(popl):
        xx = []
        for i in range(nBits):
            xx.append(Init_popl[i][j] * (2 ** i))
        XX = XL +( (XU - XL) / ((2**nBits)-1) ) * sum(xx)
        X.append(XX)
    return(X)
        
####==== generating the inital Polpulation
Init_X1_popl  = pd.DataFrame(CreatePopulation(popl, nBits))

####==== Applying the bin2dec function
X_val = bin2dec(Init_X1_popl, nBits, popl)
fit_func1 = fun1(X_val)
fit_func2 = fun1(X_val)

func_val = list(zip(fit_func1, fit_func2))
#func_val = pd.DataFrame(func_val)
#func_val.columns = ['fit_func1', 'fit_func2']

domni = []
rank1 = []
non_domni = []

for i in np.arange(popl - 1):
    if func_val[i] < func_val[i+1] :
        non_domni.append(func_val[i+1])
    else:
        domni.append(func_val[i])
    rank1 = non_domni



'''
for i in range(20):
    Init_X1_popl = Init_X1_popl
    Init_X2_popl = Init_X2_popl
    #Fitness value calculation
    X1 = [] #value of X
    X2 = []
    fit_func1 = [] #fitness
    fit_func2 = []
    for j in range(popl):   
        xx1 = []
        xx2 = []
        for i in range(nBits):
            xx1.append(Init_X1_popl[i][j] * (2 ** i))
            xx2.append(Init_X2_popl[i][j] * (2 ** i))
        XX1 = XL +( (XU - XL) / ((2**nBits)-1) ) * sum(xx1)
        XX2 = XL +( (XU - XL) / ((2**nBits)-1) ) * sum(xx2)
        X1.append(XX1)
        X2.append(XX2)
        fit_func1.append(fun1(XX1))
        fit_func2.append(fun2(XX2))
    a1 = fit_func1.index(min(fit_func1))
    a2 = fit_func2.index(min(fit_func2))
    print("X for function 1= {}".format(X1[a1]))
    print("X for function 2= {}".format(X2[a2]))
    if min(fit_func1) == 0 or (fit_func2) == 0: #Termination condition
        break
'''