#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas
from numpy.linalg import norm
from numpy import array, average, mean
import matplotlib.pyplot as plt
from random import uniform
from pprint import pp
from copy import copy
import numpy


# Read in the dataset from a CSV as a Pandas DataFrame

# In[51]:


test_dataset = pandas.read_csv('10clusters.csv',sep=',')
test_dataset = pandas.DataFrame(data=test_dataset[['X','Y']])


# Define the ranges that K can take

# In[52]:


k_values = range(2,20)


# This is just a neat visualization

# In[53]:


graph = {'x' : test_dataset['X'], 'y' : test_dataset['Y']}
plt.scatter(graph['x'],graph['y'])


# Helper function for k-means which takes a point and a dictionary of cluster centers -> cluster centroid. This computes the closest cluster from a given point.

# In[5]:


def closest_cluster(point,cluster_centers):
    min_cluster = cluster_centers[0]
    min_dist = norm(point - cluster_centers[0])

    for cluster in cluster_centers:
        dist = norm(point - cluster)

        if dist < min_dist:
            min_cluster = cluster
            min_dist = dist
    return min_cluster


def calc_center_of_masses(point_to_cluster,dataset):
    new_cluster_centers = {}
    for point in point_to_cluster:
        point = tuple(point)
        input(point)
        cluster = point_to_cluster[point]
        input(cluster)
        try:
            new_cluster_centers[cluster].append(mean(point,axis=0))
        except KeyError:
            new_cluster_centers[cluster] = [mean(point,axis=0)]
    input({cluster : mean(new_cluster_centers[cluster],axis=0)})
    return {cluster : mean(new_cluster_centers[cluster],axis=0)}

def find_bounds(dataset):
    # init the bounds
    bounds = {}
    # for all dimensions of our dataset, find the min and max
    for dimension in dataset:
        dimension = dimension.strip()
        bounds[dimension] = {'min' : dataset[dimension][0], 'max' : dataset[dimension][0]}
        # find simple min and max
        for point in dataset[dimension]:
            if point < bounds[dimension]['min']:
                bounds[dimension]['min'] = point
            elif point < bounds[dimension]['max']:
                bounds[dimension]['max'] = point
    return bounds

def create_cluster_centers(bounds,k):
    clusters = list()
    for _ in range(k):
        # fill each dimension with a legal (within bounds) value
        center_point = [uniform(bounds[dimension]['min'],bounds[dimension]['max']) for dimension in bounds]
        clusters.append(array(center_point))
    return clusters


# Here is the implemenation of k-means. It takes a K-value and returns a dictionary mapping cluster centers to the list of points that it contains,the best average_distance to the cluster centers of all points, and the average across all iterations. 'iters' number of random centers are run for each k value

# In[47]:


def k_cluster(data,k=10,precision=.01,iters=100):
    # cast DataFrame into a numpy array
    dataset = data.to_numpy()

    # Create a dict mapping 'min' and 'max' vals to each dimesion in dataset
    bounds = find_bounds(data)

    # values we will need to keep track of
    metrics = {'best_clusters' : None, 'best_dist' : 100000, 'avg_dist' : 0}

    for _ in range(iters):
        point_to_cluster = {}

        # Create k clusters randomly within the bounds
        cluster_centers = create_cluster_centers(bounds,k)

        # find the initial closest cluster to each point
        for row in dataset:
            point = tuple(row)
            point_to_cluster[point] = closest_cluster(point, cluster_centers)

        # The 'do-while' problem. This has to exist before we begin the loop
        new_cluster_centers = cluster_centers
        dist = 1

        # Keep running the algorithm until the sum of our cluster_centers falls within 'precision' Euclidean Distance of the mean of its points
        while dist > precision:

            # Save the old centers and recalculate the new centers based off of the center of mass of the points in each cluster
            old_cluster_centers = copy(new_cluster_centers)
            new_cluster_centers = calc_center_of_masses(point_to_cluster,dataset)


            # Reassign the points given the new cluster centers
            point_to_cluster = [closest_cluster(p,new_cluster_centers) for p in point_to_cluster]



            cluster_points = {key : [] for key in new_cluster_center}
            for row in data.iterrows():
                point = tuple(row[1:])
                CoM = new_cluster_center
                closest = closest_cluster(row[1:],new_cluster_center)
                cluster_points[closest].append(row[1:])
                points.append(point)
                point_to_cluster.append(closest)

        # Check the total Euclidean Distance the clusters have moved
            dist = 0
            for cluster in new_cluster_center:
                dist += norm(array(old_cluster_center[cluster]) - array(new_cluster_center[cluster]))


        # Calculate the avg distance away from each cluster
        average_dist = 0
        for i,point in enumerate(point_to_cluster):
            average_dist += abs(norm(array(point) - array(points[i])))
        average_dist = average_dist / len(point_to_cluster)
        avg_dist += average_dist
        if average_dist < best_avg_dist:
            best_avg_dist = average_dist
            best_clustering = cluster_points
    return best_clustering, best_avg_dist, avg_dist / iters


# here, we run actually run Kmeans 'k_values' times. The algorithm paramaters can be tweaked, but as per instructions, k == 2..20 and iters == 100

# In[ ]:


k_list = [[],[],[]]
for k_val in k_values:
    cluster_dict, best_dist,avg_dist = k_cluster(test_dataset,k=k_val,iters=100)
    k_list[0].append(k_val)
    k_list[1].append(avg_dist)
    k_list[2].append(best_dist)
    x_lists = []
    y_lists = []
    for cluster in cluster_dict:
        x_vals = []
        y_vals = []

        for point in cluster_dict[cluster]:
            x_vals.append(array(point)[0][0])
            y_vals.append(array(point)[0][1])
        x_lists.append(x_vals)
        y_lists.append(y_vals)


# In[ ]:


# uncommenting will plot all of the best clusters for a k
#plt.scatter(x_vals,y_vals)
    #plt.show()


# Now we plot the statistics for our run of the model

# In[54]:


# This plots the best average distance for a k
plt.scatter(k_list[0],k_list[1])
# This plots the mean average distance for a k
plt.scatter(k_list[0],k_list[2])
