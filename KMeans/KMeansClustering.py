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



# Helper function for k-means which takes a point and a dictionary of cluster centers -> cluster centroid. This computes the closest cluster from a given point.

# In[5]:


def closest_cluster(point,cluster_centers):
    min_cluster = list(cluster_centers.keys())[0]
    min_dist = norm(array(point) - norm(cluster_centers[0]))

    for cluster in cluster_centers:
        dist = norm(array(point) - array(cluster))

        if dist < min_dist:
            min_cluster = cluster
            min_dist = dist
    return min_cluster

def calc_center_of_masses(cluster_dict):
    new_cluster_centers = {cluster : 0 for cluster in cluster_dict}

    for cluster in cluster_dict:
        point_array = cluster_dict[cluster]
        cluster_center = tuple(mean(point_array,axis=0))
        new_cluster_centers[cluster] = cluster_center

    return new_cluster_centers

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
            elif point > bounds[dimension]['max']:
                bounds[dimension]['max'] = point
    return bounds

def create_cluster_centers(bounds,k):
    clusters = {k : (0,0) for k in range(k)}
    for _ in range(k):
        # fill each dimension with a legal (within bounds) value
        center_point = [uniform(bounds[dimension]['min'],bounds[dimension]['max']) for dimension in bounds]
        clusters[_] = tuple(center_point)
    return clusters

def assign_points_to_cluster(points,cluster_centers):
    p_to_c = {}
    c_to_p = {}

    for p in points:
        point = tuple(p)
        cluster = closest_cluster(point, cluster_centers)

        p_to_c[point] = cluster

        try:
            c_to_p[cluster].append(point)
        except KeyError:
            c_to_p[cluster] = [point]

    return p_to_c, c_to_p
# Here is the implemenation of k-means. It takes a K-value and returns a dictionary mapping cluster centers to the list of points that it contains,the best average_distance to the cluster centers of all points, and the average across all iterations. 'iters' number of random centers are run for each k value

# In[47]:
def k_cluster(data,k=10,precision=.1,iters=100):
    # cast DataFrame into a numpy array
    dataset = data.to_numpy()

    # Create a dict mapping 'min' and 'max' vals to each dimesion in dataset
    bounds = find_bounds(data)
    # values we will need to keep track of
    metrics = {'best_clusters' : None, 'best_avg_dist' : 100000, 'overall_avg_dist' : 0}
    points = {'vals' : [tuple(row) for row in dataset], 'to_cluster' : {}}
    clusters = {'centers' : {}, 'to_points' : {}, 'centers_updated' : {}}

    for _ in range(iters):

        # Create k clusters randomly within the bounds
        clusters['centers'] = create_cluster_centers(bounds,k)

        points['to_cluster'], clusters['to_points'] = assign_points_to_cluster(points['vals'],clusters['centers'])

        # The 'do-while' problem. This has to exist before we begin the loop
        clusters['centers_updated'] = clusters['centers']
        dist = 1

        # Keep running the algorithm until the sum of our cluster_centers falls within 'precision' Euclidean Distance of the mean of its points
        while dist > precision:

            # Save the old centers and recalculate the new centers based off of the center of mass of the points in each cluster
            clusters['centers'] = copy(clusters['centers_updated'])

            clusters['centers_updated'] = calc_center_of_masses(clusters['to_points'])

            # Reassign the points given the new cluster centers
            points['to_cluster'], clusters['to_points'] = assign_points_to_cluster(points['vals'],clusters['centers_updated'])

            # Check the total Euclidean Distance the clusters have moved
            dist = 0
            for cluster in clusters['centers_updated']:
                try:
                    dist += abs(norm(array(clusters['centers'][cluster]) - array(clusters['centers_updated'][cluster])))
                except KeyError as a:
                    print(f"bad key: {a}")

        # Calculate the avg distance away from each cluster
        average_dist = 0
        n = len(points['vals'])

        for point,cluster in points['to_cluster'].items():
            average_dist += abs(norm(array(point) - array(clusters['centers_updated'][cluster])))

        average_dist = average_dist / n
        metrics['overall_avg_dist'] += average_dist

        if average_dist < metrics['best_avg_dist']:
            metrics['best_avg_dist'] = average_dist
            metrics['best_clusters'] = points['to_cluster']

    return metrics['best_clusters'], metrics['best_avg_dist'], metrics['overall_avg_dist'] / iters


# here, we run actually run Kmeans 'k_values' times. The algorithm paramaters can be tweaked, but as per instructions, k == 2..20 and iters == 100

# In[ ]:


k_list = [[],[],[]]
for k_val in k_values:
    p_to_c, best_avg_dist, avg_dist = k_cluster(test_dataset,k=k_val,iters=10)
    k_list[0].append(k_val)
    k_list[1].append(avg_dist)
    k_list[2].append(best_avg_dist)

pp(k_list[1])
pp(k_list[2])
plt.scatter(k_list[0],k_list[1])
plt.scatter(k_list[0],k_list[2])
plt.xticks([x for x in range(2,20)])
plt.show()
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
