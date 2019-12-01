# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:32:31 2019
@author: Salam Saudagar
"""

import numpy as np, math

dist_vec = np.random.uniform(0,100,6)

dis_mat = np.array([[0,dist_vec[0],dist_vec[1],dist_vec[2]],
                    [dist_vec[0],0,dist_vec[3],dist_vec[4]],
                    [dist_vec[1],dist_vec[3],0,dist_vec[5]],
                    [dist_vec[2],dist_vec[4],dist_vec[5],0]])

desire_mat = 1 / dis_mat
desire_mat[desire_mat == math.inf] = 0

ph_vec = np.random.uniform(0.1,1,6)

pheromone_mat = np.array([[0,ph_vec[0],ph_vec[1],ph_vec[2]],
                          [ph_vec[0],0,ph_vec[3],ph_vec[4]],
                          [ph_vec[1],ph_vec[3],0,ph_vec[5]],
                          [ph_vec[2],ph_vec[4],ph_vec[5],0]])

alpha = 1.1 #pheromone factor 
beta = 1.2 #visibility factor
ant_count = 4
city_count = 4
trans_coef = 0.7
max_iter = 50

closeness_matrix = pheromone_mat*np.power(desire_mat,beta)
pheromone_fac = np.power(pheromone_mat,alpha)
inv_dist_fac  = np.power(desire_mat,beta)
route = np.ones((4,5))
dist_val = []

for x in range(ant_count):
        cities = []
        start = int(np.random.choice([0,1,2,3],1))
        closeness_mat_rule_1 = pheromone_mat*inv_dist_fac
        probval = np.multiply(pheromone_fac,inv_dist_fac)
        prob_decide = probval / sum(sum(probval)) 
        for y in range(city_count - 1):
            current_trans_coef = np.random.uniform(0,1)
            print(current_trans_coef)
            if current_trans_coef <= trans_coef:
                print('exploitation')
                cities.append(np.argmax(closeness_mat_rule_1[start,:]))
                closeness_mat_rule_1[start,np.argmax(closeness_mat_rule_1[start,:])] = 0
                prob_decide[start,np.argmax(prob_decide[start,:])] = 0
            else:
                print('exploration')
                cities.append(np.argmax(prob_decide[start,:]))
                closeness_mat_rule_1[start,np.argmax(closeness_mat_rule_1[start,:])] = 0
                prob_decide[start,np.argmax(prob_decide[start,:])] = 0
        route[x,0] = start
        route[x,4] = start
        route[x,1] = cities[0]
        route[x,2] = cities[1]
        route[x,3] = cities[2]
        dist_val.append(dis_mat[start,cities[0]] + dis_mat[cities[0],cities[1]] + 
                        dis_mat[cities[1],cities[2]] + dis_mat[cities[2],start])