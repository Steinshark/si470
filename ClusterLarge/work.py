import numpy as np
import pandas as pd
from scipy.sparse import *
from sklearn.decomposition import PCA

path        = r'/home/mids/m226252/si470/data'
docNYT      = fr'{path}/docword.nytimes.txt'
vocabNYT    = fr'{path}/vocab.nytimes.txt'

def read_data_to_df(filename,header=3):
    with open(filename,'r') as file:
        n_articles      = int(file.readline())
        n_words         = int(file.readline())
        n_words_total   = int(file.readline())

        vocab = {}
        with open(vocabNYT,'r') as vocab_file:
            for i, word in enumerate(vocab_file.readlines()):
                vocab[i] = word

        # define the size of the dataset we will build
        rows = n_words
        cols = n_articles
        matrix = csr_matrix((rows,cols), dtype = np.float64)

        for 
        print(matrix[0,1])



read_data_to_df(docNYT,header=3)
