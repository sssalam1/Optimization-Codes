# Random Search Method
"""
Created on Thu Aug  8 22:45:55 2019
@author: Salam Saudagar
"""
import numpy as np
# import random as rd
import pandas as pd
import matplotlib.pyplot as plt

def Rosenbrock(x,y, a=1, b=100):
    return( (a-x)**2 + b*(y - x**2 )**2 )
    
x_initial, y_initial = np.random.uniform(-5,5, size = 2)
EN_int = Rosenbrock(x_initial,y_initial)

x_vec , y_vec , EN = [] , [] , []

for i in range(500):
    h1 , h2 = np.random.uniform(-1,1, size = 2)
    x_new,y_new = x_initial + h1, y_initial + h2
    EN_new = Rosenbrock(x_new , y_new)
    x_vec.append(x_new)
    y_vec.append(y_new)
    EN.append(EN_new)
    
tab = pd.concat([pd.DataFrame(x_vec),pd.DataFrame(y_vec),pd.DataFrame(EN)] , axis = 1)
tab.columns=["x","y","Energy"]    
plt.plot(tab["Energy"])

'''
for i in range(0 , 500):
    h = rd.uniform(-1,1)
    k = Rosenbrock(x + h, y + h)
    r.append(x + h)
    s.append(y + h)
    t.append(k)
    
tab = pd.concat([pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(t)] , axis = 1)
tab.columns=["x","y","fx"]    

plt.plot(tab["fx"])
'''