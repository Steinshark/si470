import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
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
    partial_ratings_matrix  = small_ratings_matrix.to_numpy()
    x,y = np.nonzero(partial_ratings_matrix)
    filter_matrix = scipy.sparse.csr_matrix(([1 for _ in x], zip(x,y)),shape=np.shape(partial_ratings_matrix)).to_dense()
    
    print()
    # convert all non_zero elements to 1

    printc(f"\tRead data in {(time()-t1):.3f} seconds",GREEN)
    return partial_ratings_matrix, filter_matrix

ratings, filter_matrix = read_data()
Algorithms.GradientDescent_optimized(ratings,filter_matrix,dim=100,alpha=.01,iters=10)
