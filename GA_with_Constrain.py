# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:46:37 2019

@author: Salam Saudagar
"""

import numpy as np
import pandas as pd

bits  = 4
cross_prob = 0.7
mute_prob = 0.3

def finess_function(x,y):
    fx = (1 - x)**2 + 100*(y - x**2)**2
    gx1 = x*y + x - y + 1.5
    gx2 = 10 - x*y
    r = np.linspace(0.2,1,5).tolist()
    fitness = []
    for i in r:
        fit = fx + i * max(0,gx1)**2 + i * max(0,gx2)**2
        fitness.append(fit)
    fitnss = np.min(fitness)                  # minimization problem
    return fitnss
def bin_to_dec(binary1):
    decimal = 0
    for digit in binary1:
        decimal = decimal*2 + int(digit)
    return decimal


def binary_generation(pop_size):
    a = []
    b = []
    for i in range(10):
        binary = np.random.choice([0,1],size = bits)
        decimal = bin_to_dec(binary)
        b.append(decimal)
        a.append(binary)
    return a,b

x1_bit,x1_dec = binary_generation(10)
x2_bit,x2_dec = binary_generation(10)


#======================Calculate fitness value================================#

optimum_num = []
for i in range(500):
    XL1 = 0 ; XU1 = 1
    XL2 = 0 ; XU2 = 13
    fitness = []
    d1 = []
    d2 = []
    for i in range(10):
        D1 = XL1+((XU1-XL1)/float((2**bits)-1)) * x1_dec[i]
        D2 = XL2+((XU2-XL2)/float((2**bits)-1)) * x2_dec[i]
        d1.append(D1)
        d2.append(D2)
        fit = finess_function(D1,D2)
        fitness.append(fit)
    min_value = np.argmin(fitness)
    function_value = min(fitness)
    x_val = x1_dec[min_value]
    y_val = x2_dec[min_value]
    optimal = [x_val,y_val,function_value]
    optimum_num.append(optimal)
    fitness_binary = []
    for i in range(10):
        combine_bin = np.concatenate([x1_bit[i],x2_bit[i]])
        combine_bin1 = np.append(combine_bin,fitness[i])
        fitness_binary.append(combine_bin1)
    selection_data = pd.DataFrame(fitness_binary)
    #===============================Tournament selection=================#
    select_rand = np.random.randint(0,10,size=(10,2))
    tourn_select = []
    for l in range(len(select_rand)):
        select1 = select_rand[l][0]
        select2 = select_rand[l][1]
        sel1_fit1 = selection_data.iloc[select1,8]
        sel1_fit2 = selection_data.iloc[select2,8]
        if sel1_fit1 < sel1_fit2:
            tourn_select.append(selection_data.iloc[select1])
        else:
            tourn_select.append(selection_data.iloc[select2])
    tourn_df = pd.DataFrame(np.array(tourn_select))
    tourn_df = tourn_df.copy()
    tourn_df1 = tourn_df.drop(8,axis=1).astype(int)
    #=================Cross-over==========================================#
    cross_rand = np.random.randint(0,10,size=(5,2))
    rnd_no = np.random.uniform(0,1)
    cross_prob = 0.6
    cross_df = []
    for m in range((tourn_df1.shape[0])//2):
        cross1 = cross_rand[m][0]
        cross2 = cross_rand[m][1]
        cross_fit1 = tourn_df1.iloc[cross1]
        cross_fit2 = tourn_df1.iloc[cross2]
        for i in range(tourn_df1.shape[1]):
            rnd_no = np.random.uniform(0,1)
            if rnd_no < cross_prob:
                break
        cross_ex1 = cross_fit1.iloc[0:i,].append(cross_fit2.iloc[i:,])
        cross_ex2 = cross_fit2.iloc[0:i,].append(cross_fit1.iloc[i:,])
        cross_df.append(cross_ex1)
        cross_df.append(cross_ex2)
    cross_feature = pd.DataFrame(np.array(cross_df))
    cross_feature1 = cross_feature.copy()
    #=============mutation==============================================#
    for i in range(cross_feature.shape[0]):
        mute_x = cross_feature.loc[i]
        for j in range(cross_feature.shape[1]):
            rnd_no = np.random.uniform(0,1)
            if rnd_no < mute_prob:
                if mute_x[j] == 0:
                    mute_x[j] = 1
                    break
                else:
                    mute_x[j] = 0
                    break                  
    new_data = cross_feature.copy()
    #=====================bin and decilamal conversion==================#
    x1_bit = []
    x2_bit = []
    x1_dec = []
    x2_dec = []
    for i in range(len(new_data)):
        binary1 = list(new_data.loc[i][0:4])
        binary2 = list(new_data.loc[i][4::])
        x1_bit.append(binary1)
        x2_bit.append(binary2)
        decimal1 = bin_to_dec(binary1)
        decimal2 = bin_to_dec(binary2)
        x1_dec.append(decimal1)
        x2_dec.append(decimal2)
    
optimal_fitness = np.argmin(fitness)  
print("minimum fitness value is %f" %optimal_fitness)
print("optimal x and y value is %f and %f" %(d1[optimal_fitness],d2[optimal_fitness]))