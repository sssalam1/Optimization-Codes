# MonteCarlo Method
"""
Created on Thu Aug  8 22:45:55 2019
@author: Salam Saudagar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Rosenbrock(x,y, a=1, b=100):
    return( (a-x)**2 + b*(y - x**2 )**2 )
    
x_initial, y_initial = np.random.uniform(-5.5, size  = 2)
EN_initial = Rosenbrock(x_initial,y_initial)

x_vec , y_vec , EN_vec= [] , [] , []

for i in range(0 , 5000):
    h1, h2 = np.random.uniform(-1,1, size = 2)
    x_new, y_new = x_initial + h1, y_initial + h2
    EN_new = Rosenbrock(x_new , y_new)
    Del_E = EN_new - EN_initial
    if Del_E  < 0:
        x_initial , y_initial , EN_initial = x_new, y_new , EN_new
        x_vec.append(x_new)
        y_vec.append(y_new)
        EN_vec.append(EN_new)
    else:
        # x_initial , y_initial , EN_initial = x_initial , y_initial , EN_initial
        x_vec.append(x_initial)
        y_vec.append(y_initial)
        EN_vec.append(EN_initial)
            
tab = pd.concat([pd.DataFrame(x_vec),pd.DataFrame(y_vec),pd.DataFrame(EN_vec)] , axis = 1)
tab.columns=["x","y","Energy"]    
#plt.plot(tab["Energy"])
min_i = np.argmin(EN_vec) #Calculating the index at which the energy is Minimum
print("The Optimized Value of x and y are: {} and {} respectively ".format(x_vec[min_i], y_vec[min_i]))
plt.plot(x_vec,EN_vec,'-r')
plt.plot(y_vec,EN_vec,'-g')