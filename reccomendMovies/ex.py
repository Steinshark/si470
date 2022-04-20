import sys
import pandas as pd
import numpy as np

from time import time
from os.path import join
sys.path.append(join('D:classes'))

if sys.argv[1] == 'GPU':
    from cupy.sparse import lil_matrix
else:
    from scipy.sparse import lil_matrix



#init all of the filenames and dataframes
datasets = {    'small'     :  {'movies' : join('ml-latest-small','movies.csv') , 'ratings' : join("ml-latest-small","ratings.csv") , 'tags' : join("ml-latest-small","tags.csv")},
                'large'     :  {'movies' : join("ml-latest","movies.csv")       , 'ratings' : join("ml-latest","ratings.csv")       , 'tags' : join('ml-latest',"tags.csv")}}
dataframes= {   'small'     :  {'movies' : None, 'ratings' :  None, 'tags' : None},
                'large'     :  {'movies' : None, 'ratings' :  None, 'tags' : None}}


################################################################################
#                           READ IN DATASETS
#           ASSUMPTION TO BE MADE HERE: how do we fill the na?
################################################################################

t1 = time()
for size in dataframes:
    for type in dataframes[size]:
        dataframes[size][type] = pd.read_csv(datasets[size][type],sep=',')

################################################################################
#                           GET SIZE DATA
################################################################################

n_movies = dataframes['large']['movies']['movieId'].iloc[-1]
n_users  = dataframes['large']['ratings']['userId'].iloc[-1]
print(f"shape: {n_users,n_movies}")

################################################################################ .08
#                        INIT MATRIX FOR READ
################################################################################ 2.2

lil_matr = lil_matrix((n_users+1,n_movies+1))
file = open(datasets['large']['ratings'])
file.readline()# Remove fields from top of file

################################################################################
#                        READ FILE
################################################################################
for line in file:
    userId, movieId, ratingId, timestamp = line.split(',')
    lil_matr[int(userId),int(movieId)] = float(ratingId)

################################################################################
#                        INIT MATRIX FOR READ
################################################################################

print(f"\tsize: {lil_matr.data.size/(1024**2):.2f} MB")
print(f'finished in {time()-t1} seconds')
