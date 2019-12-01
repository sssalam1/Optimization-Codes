# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:12:28 2019

@author: Salam Saudagar
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
da = load_iris()

data = pd.DataFrame(da.data)
data.columns = da.feature_names
label = da.target

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)

q0 = 0.7
subset = 3
ants = 4
max_iter = 50
phe = np.random.uniform(0.1,1,4)

for m in range(max_iter):
    #### Calculating the path for each ants
    path =[]
    for j in np.arange(ants):
        new_ph =np.copy(phe)
        cityA=[]
        A = np.random.randint(0,3)
        cityA.append(A)
        for i in np.arange(subset-1):
            r = np.random.uniform(0,1)        
            if r < q0:
                new_ph[A]=0
                A = np.argmax(new_ph)
                cityA.append(A)
            else:
                new_ph[A]=0
                prob = new_ph / (sum(new_ph))
                np.random.choice(range(4),p=prob)
                cityA.append(A)     
        path.append(cityA)  
        
    #### Calculating the Fitness of each path by using SVM
    accuracy=[]
    for k in np.arange(len(path)):
        model = SVC(random_state=42)
        model.fit( X_train.iloc[:,path[k]],y_train)
        y_pred = model.predict(X_test.iloc[:,path[k]])
        acc= accuracy_score(y_test,y_pred)
        accuracy.append(acc)  
    max_acc = np.max(accuracy)   
    max_acc_ind = np.argmax(accuracy)
    best_atts = path[max_acc_ind]
    
    #### Changing the Pheromene concentration
    phe = phe* 0.8
    for l in np.arange(len(best_atts)):
        q= (phe[best_atts[l]]*1.2) / 0.8
        phe[best_atts[l]] =q 
    final_Attribtes = ((np.argsort(phe).tolist())[::-1])[:subset]

print("The Best Attributes are:",final_Attribtes)
print("The Best accuracy is:",max_acc*100)