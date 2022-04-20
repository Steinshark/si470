import tensorflow as tf
import pandas as pd
import numpy as np
import os
import Algorithms
from terminal import *
from time import time

# For s**ts and giggles
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.run_functions_eagerly(True)

#TYPES
_f32            = tf.dtypes.float32
_i32            = tf.dtypes.int32
#DATASET LOCATIONS
_RATINGS_SMALL  = os.path.join("ml-latest-small","ratings.csv")
_MOVIE_ID_SMALL = os.path.join("ml-latest-small","movies.csv")
_HEADERS        = ["Black Panther",
                   "Pitch Perfect",
                   "Star Wars: The Last Jedi",
                   "It",
                   "The Big Sick",
                   "Lady Bird",
                   "Pirates of the Caribbean",
                   "Despicable Me",
                   "Coco",
                   "John Wick",
                   "Mamma Mia",
                   "Crazy Rich Asians",
                   "Three Billboards Outside Ebbings, Missouri",
                   "The Incredibles"]

_MOVIE_ID_MAP      = {}
_ROW_MAP           = {}


def fill(df,method):
    for col in df:
        if method == 'mean':
            replacement = df[col].mean()
        elif method   == 'zero':
            replacement = 0
        df[col].fillna(value = replacement, inplace = True)


def read_data():
    printc(f"\tReading data",BLUE)
    t1 = time()

    # Read 'ratings.csv' into a DataFrame from the small set
    small_ratings   = pd.read_csv(  _RATINGS_SMALL,
                                    dtype = {   'userId'    :   np.float32,
                                                'movieId'   :   np.float32,
                                                'rating'    :   np.float32,
                                                'timestamp' :   np.float32})

    small_movie_ids  = pd.read_csv(  _MOVIE_ID_SMALL,
                                    dtype = {   'movieId' : np.float32})

    # Convert the dataframe to correct matrix format: (rows=MovieId, cols=userId)
    small_ratings_matrix = small_ratings.pivot( index='movieId',    columns='userId',   values='rating')

    # Map movieId to row and vice versa
    for row, id in enumerate(small_movie_ids['movieId']):
        _MOVIE_ID_MAP[row]  = id
        _ROW_MAP[id]        = row
    #info
    # Fill with our choice
    fill(small_ratings_matrix,'zero')

    # convert the dataframe to a Tensor
    partial_ratings_tensor  = tf.convert_to_tensor(  small_ratings_matrix,   dtype = _f32)

    # convert all non_zero elements to 1
    partial_ratings_filter  = tf.sparse.to_dense(tf.sparse.map_values(tf.ones_like,tf.sparse.from_dense(partial_ratings_tensor)))
    partial_ratings_filter  = tf.cast(partial_ratings_filter, dtype=_f32)

    printc(f"\tRead data in {(time()-t1):.3f} seconds",GREEN)
    return partial_ratings_tensor, partial_ratings_filter

ratings, filter_matrix = read_data()
# Ascertain what dimensions we are in
rows    = tf.Variable(ratings.shape[0],dtype=tf.dtypes.int32)
cols    = tf.Variable(ratings.shape[1],dtype=tf.dtypes.int32)

alphas = [.001,.0001]
iter_val = [1,3,5]
dimensions = [40,50]


runs =      {a : { d : {'time' : [], 'rmse' : []} for d in dimensions} for a in alphas}
colors =    {.00001 : 'k', .5 : 'g', .000001 : 'r', .05 : 'm', .01 : 'c', .001 : 'b', .0001 : 'y'}

#for d in dimensions:
#    printc(f"\nDIMENSION: {d}",BLUE)
#    for a in alphas:
#        printc(f"\nGathering data for alpha = {a}\n\titer= ",TAN,endl='')
#        for test_val in iter_val:
#            lowest_score = 100000000
#            t1 = time()
#            for i in range(1):
#                p,q = Algorithms.GradientDescent_optimized(ratings,filter_matrix,dim=d,iters=test_val,alpha = a)
#                run_err = Algorithms.RMSE(ratings,filter_matrix,p,q)
#                if run_err < lowest_score:
#                    lowest_score = run_err
#            printc(f" {test_val}:{(time()-t1):.2f}  ",TAN,endl='')
#            runs[a][d]['rmse'].append(lowest_score)

p,q,errors = Algorithms.GradientDescent_optimized(ratings,filter_matrix,dim=20,iters=10,alpha = .001)


plt.plot(errors)

#from matplotlib import pyplot as plt
#fig, ax = plt.subplots(len(dimensions))
#
#for i,d in enumerate(dimensions):
#    for r,a in enumerate(runs):
#        ax[i].set_title(f"dim = {dimensions[i]}",fontsize=10)
#        ax[i].plot(iter_val,runs[a][d]['rmse'], color=f"{colors[a]}",marker='x',label=f"a={a}_rmse")
#        ax[i].set_ylabel("rmse")
#        ax[i].set_xlabel("iterations run")
#plt.legend()
#
#plt.show()
