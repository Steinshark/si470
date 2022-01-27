#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas 
from numpy.linalg import norm
from numpy import array, average, mean
import matplotlib.pyplot as plt  
from random import uniform
from pprint import pp
from copy import copy


# In[25]:


test_dataset = pandas.read_csv('10clusters.csv',sep=',')
test_dataset = pandas.DataFrame(data=test_dataset[['X','Y']])


# In[26]:


#Build the k values that will be tested over 
k_values = range(2,20)


# In[27]:


graph = {'x' : test_dataset['X'], 'y' : test_dataset['Y']}
plt.scatter(graph['x'],graph['y'])


# In[28]:


def closest_cluster(point,cluster_centers):   
    min_cluster = list(cluster_centers.keys())[1]
    min_dist = norm(array(point) - array(cluster_centers[min_cluster]))
    for cluster in cluster_centers:
        dist = norm(array(point) - array(cluster_centers[cluster]))

        if dist < min_dist:
            min_cluster = cluster
            min_dist = dist
    return min_cluster


# In[29]:


def calc_center_of_masses(cluster_points):
    new_com = {}
    for cluster in cluster_points:
        com = mean(cluster_points[cluster],axis=0)
        try:
            new_com[cluster] = com[0]
        except IndexError:
            print(f"strange: {com}")
            new_com[cluster] = com
    return new_com


# In[30]:


def k_cluster(data,k=10):
    
    # Find bounds for placing the random cluster centers 
    bounds = {}
    for column in data[1:]:
        column = column.strip()
        bounds[column] = {'low' : data[column][0],'high' : data[column][0]}
        for point in data[column][1:]:
            if point < bounds[column]['low']:
                bounds[column]['low'] = point
            elif point > bounds[column]['high']:
                bounds[column]['high'] = point
                
    # place random cluster centers
    cluster_center = {}
    for cluster_number in range(k):
        
        cluster_center[cluster_number] = [uniform(bounds[dimension]['low'],bounds[dimension]['high']) for dimension in bounds]
    cluster_points = {cluster  :[] for cluster in cluster_center}
    
    for row in data.iterrows():
        cluster_points[closest_cluster(row[1:],cluster_center)].append(row[1:])
    old_cluster_center = None
    dist = 1
    while dist > .01:
        dist = 0
        # shift the cluster masses
        if old_cluster_center is None:
            old_cluster_center = copy(cluster_center)
        else:
            old_cluster_center = copy(new_cluster_center)
        new_cluster_center = calc_center_of_masses(cluster_points)

        # Reassign the datapoints to the clusters 
        cluster_points = {key : [] for key in new_cluster_center}
        for row in data.iterrows():
            point = row[1:]
            com = new_cluster_center
            closest = closest_cluster(row[1:],new_cluster_center)
            
            cluster_points[closest].append(row[1:])
            
        # Check the distance 
        for cluster in new_cluster_center:
            dist += norm(array(old_cluster_center[cluster]) - array(new_cluster_center[cluster]))

        
    return cluster_points


# In[31]:


get_ipython().run_cell_magic('time', '', 'for k_val in k_values:\n    centers = k_cluster(test_dataset,k=k_val)\n\n    x_lists = []\n    y_lists = []\n\n    for cluster in centers:\n        x_vals = []\n        y_vals = []\n\n        for point in centers[cluster]:\n            x_vals.append(array(point)[0][0])\n            y_vals.append(array(point)[0][1])\n        x_lists.append(x_vals)\n        y_lists.append(y_vals)\n    \n #       for i in range(len(x_lists)):\n #           plt.scatter(x_lists[i],y_lists[i])\n #   plt.show()\n')


# In[24]:


f = open("10Clusters.csv",'w')
from random import uniform, randint
f.write(F"X,Y\n")
bounds = [-100,100]
clusters = [(uniform(bounds[0],bounds[1]),(uniform(bounds[0],bounds[1]))) for i in range(5)]

print(clusters)
for i in range(1000):
    cl = clusters[randint(0,len(clusters)-1)]
    x = uniform(cl[0]-(bounds[0]/50),cl[1]-(bounds[1]/55))
    y = uniform(cl[0]-(bounds[0]/50),cl[1]-(bounds[1]/55))
    f.write(f"{x},{y}\n")
f.close()
    


# In[ ]:




