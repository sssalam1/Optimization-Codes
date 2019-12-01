"""
Created on Mon Sep 16 14:06:29 2019
@author: Salam Saudagar

Genetic Algorithm: static and dynamic Constraints
"""
import numpy as np
import pandas as pd

# Objective function
def function(x1,x2):
    return(100 * (x1**2 - x2)**2 + (1 - x1)**2)
    
nBit = 6
XU,XL,YU,YL = 50,-50,50,-50
popln = 10 ;Pc = 0.7 ; Pm = 1/popln

def CreatePopulation(popln, nBit):
    popl_binary = []
    for i in range(popln):
        binr1 = np.random.choice([0, 1], size=nBit)
        popl_binary.append(binr1)
    return popl_binary

Init_X1_popl, Init_X2_popl = pd.DataFrame(CreatePopulation(popln, nBit)),pd.DataFrame(CreatePopulation(popln, nBit))
op = pd.DataFrame(columns=['X1','X2','Fitness'])
for i in range(100):
    Init_X1_popl, Init_X2_popl = Init_X1_popl, Init_X2_popl
    #Fitness value calculation
    X1, X2 = [], [] #value of X
    fit = [] #fitness
    for j in range(popln):   
        xx1,xx2 = [], []
        for i in range(nBit):
            xx1.append(Init_X1_popl[i][j] * (2 ** i))
            xx2.append(Init_X2_popl[i][j] * (2 ** i))
        XX1 = XL +( (XU - XL) / ((2**nBit)-1) ) * sum(xx1)
        XX2 = YL +( (YU - YL) / ((2**nBit)-1) ) * sum(xx2)
        if (XX1*XX2 + XX1 - XX2 + 1.5) <=0:
            X1.append(XX1)
            X2.append(XX2)
            fit.append(function(XX1,XX2))
    a1 = fit.index(min(fit))
    print("X1 = {}".format(X1[a1]))
    print("X2 = {}".format(X2[a1]))
    if min(fit) == 0: #Termination condition
        break
    #Tournament selection
    selct_chromX, selct_chromY = [], []
    for j in range(popln):
        ran1, ran2 = np.random.randint(0,9,2) #Generating random int between 0 to 9 of size 2
        n = [fit[ran1],fit[ran2]]
        if n[0] < n[1]:
            selct_chromX.append(list(Init_X1_popl.iloc[ran1]))
            selct_chromY.append(list(Init_X1_popl.iloc[ran1]))
        else:
            selct_chromX.append(list(Init_X1_popl.iloc[ran2]))
            selct_chromY.append(list(Init_X2_popl.iloc[ran2]))

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







#x1 = range(0,1,0.05)
#x2 = range(0,13,0.05)


"""
Constrained Minimization Problem

We want to minimize a simple fitness function of two variables x1 and x2

   min f(x) = 100 * (x1^2 - x2) ^2 + (1 - x1)^2;
    x

such that the following two nonlinear constraints and bounds are satisfied

   x1*x2 + x1 - x2 + 1.5 <=0, (nonlinear constraint)
   10 - x1*x2 <=0,            (nonlinear constraint)
   0 <= x1 <= 1, and          (bound)
   0 <= x2 <= 13              (bound)

"""