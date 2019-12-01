# Genetic Algorithm
"""
Created on Sat Aug 10 01:28:40 2019
@author: Salam Saudagar
"""

import numpy as np
import pandas as pd

#Rosenbrock function:
def rosen_brk(x,y):
    return(((1 - x)**2) + (100 * ((y - x**2)**2)))
    
nBit = 6
XU,XL,YU,YL = 5,-5,5,-5
popln = 10 ;Pc = 0.7 ; Pm = 1/popln

def CreatePopulation(popln, nBit):
    popl_binary = []
    for i in range(popln):
        binr1 = np.random.choice([0, 1], size=nBit)
        popl_binary.append(binr1)
    return popl_binary


Init_X_popl, Init_Y_popl = pd.DataFrame(CreatePopulation(popln, nBit)),pd.DataFrame(CreatePopulation(popln, nBit))
op = pd.DataFrame(columns=['X','Y','Fitness'])
for i in range(100):
    Init_X_popl, Init_Y_popl = Init_X_popl, Init_Y_popl
    #Fitness value calculation
    X, Y = [], [] #value of X
    fit = [] #fitness
    for j in range(popln):   
        xx,yy = [], []
        for i in range(nBit):
            xx.append(Init_X_popl[i][j] * (2 ** i))
            yy.append(Init_Y_popl[i][j] * (2 ** i))
        XX = XL +( (XU - XL) / ((2**nBit)-1) ) * sum(xx)
        YY = YL +( (YU - YL) / ((2**nBit)-1) ) * sum(yy)
        X.append(XX)
        Y.append(YY)
        fit.append(rosen_brk(XX,YY))
    a1 = fit.index(min(fit))
    print("X = {}".format(X[a1]))
    print("Y = {}".format(Y[a1]))
    if min(fit) == 0: #Termination condition
        break
    #Tournament selection
    selct_chromX, selct_chromY = [], []
    for j in range(popln):
        ran1, ran2 = np.random.randint(0,9,2) #Generating random int between 0 to 9 of size 2
        n = [fit[ran1],fit[ran2]]
        if n[0] < n[1]:
            selct_chromX.append(list(Init_X_popl.iloc[ran1]))
            selct_chromY.append(list(Init_Y_popl.iloc[ran1]))
        else:
            selct_chromX.append(list(Init_X_popl.iloc[ran2]))
            selct_chromY.append(list(Init_Y_popl.iloc[ran2]))

    selct_chromX = pd.DataFrame(selct_chromX)   
    selct_chromY = pd.DataFrame(selct_chromY)

    #Crossover
    CrossX , CrossY = [], []
    for i in range(int(popln/2)):
        ran1, ran2 = np.random.randint(0,9,2)
        crX1, crX2 = list(selct_chromX.iloc[ran1]), list(selct_chromX.iloc[ran2])
        crY1, crY2 = list(selct_chromY.iloc[ran1]), list(selct_chromY.iloc[ran2])
        for j in range(nBit):
            rn = np.random.uniform(0,1)
            if rn < Pc: 
                break
        crossX1, crossX2 = crX1[0:j] + crX2[j:], crX2[0:j] + crX1[j:]
        crossY1, crossY2 = crY1[0:j] + crY2[j:], crY2[0:j] + crY1[j:]        
        CrossX.append(crossX1), CrossX.append(crossX2)
        CrossY.append(crossY1), CrossY.append(crossY2)

    Init_X_popl = pd.DataFrame(CrossX)
    Init_Y_popl = pd.DataFrame(CrossY) 