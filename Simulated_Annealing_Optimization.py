# Simulated Annealing
"""
Created on Fri Aug  9 19:16:09 2019
@author: Salam Saudagar
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

def Rosenbrock(x,y, a=1, b=100):
    return( (a-x)**2 + b*(y - x**2 )**2 )
    
x_initial , y_initial = np.random.uniform(-5, 5, size = 2)
EN_initial = Rosenbrock(x_initial,y_initial)

###===== Creating the loop for number of steps, to calculate the Initial Temperature
x_vec , y_vec , Accept_EN_vec , Reject_EN_vec= [] , [] , [] , []
T0 = 1000 

for i in range(500):
    h1 , h2 = np.random.uniform(-1,1, size = 2)
    x_new , y_new = x_initial + h1 , y_initial + h2
    EN_new = Rosenbrock(x_new , y_new)
    Del_E = EN_new - EN_initial
    r = np.random.uniform(0,1)
    #x_vec.append(x_new)
    #y_vec.append(y_new)
    if r  < np.exp(-Del_E/T0):
        x_initial , y_initial , EN_initial = x_new , y_new , EN_new
        Accept_EN_vec.append(EN_new)
    else:
        #x_vec.append(x_initial)
        #y_vec.append(y_initial)
        Reject_EN_vec.append(EN_initial)
     
###===== Calculating the initial Temperature T0
m1 = len(Accept_EN_vec)
m2 = len(Reject_EN_vec)
del_f = np.mean(Accept_EN_vec)
 
T0 = -del_f / np.log( (0.95 *(m1 + m2) - m1) / m2 )
print("Initial Temperature = %f"%T0)

###===== Loop for Simulated Annuling, to Calculate optimum
x_initial , y_initial = np.random.uniform(-5, 5, size = 2)
EN_initial = Rosenbrock(x_initial,y_initial)

x_vec , y_vec , EN_vec = [] , [] , []

for j in np.arange(T0):
    itr = 1000
    for k in range(itr):
        h1 , h2 = np.random.uniform(-1,1, size = 2)
        x_new , y_new = x_initial + h1 , y_initial + h2
        EN_new = Rosenbrock(x_new , y_new)
        Del_E = EN_new - EN_initial
        r = np.random.uniform(0,1)
        x_vec.append(x_new)
        y_vec.append(y_new)
        
        if r  < np.exp(-Del_E/T0):
            x_initial , y_initial , EN_initial = x_new , y_new , EN_new
            EN_vec.append(EN_new)
        else:
            EN_vec.append(EN_new)
    if len(EN_vec) >= 100 or itr >= 100:
        T0 = 0.9 * T0
        if T0 <= 1:
            break

print("Final Temperature = %f"%T0)
min_i = np.argmin(EN_vec) #Calculating the index at which the energy is Minimum
x_min, y_min = x_vec[min_i], y_vec[min_i]
print("Minimum Energy is = %0.4f."%EN_vec[min_i])
print("The Optimized Value of x and y are: {} and {} respectively ".format(x_min, y_min))
#plt.plot(EN_vec)
#plt.plot(x_vec,EN_vec,'-r')
#plt.plot(y_vec,EN_vec,'-g')