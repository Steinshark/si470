import numpy as np
import pandas as pd
from time import time
from colors import *
import os
from scipy.sparse import lil_matrix, csr_matrix, load_npz, save_npz

from sklearn.decomposition import TruncatedSVD, SparsePCA

def printc(s,color,endl='\n'):
    print(f"{Color.colors[color]}{s}{Color.colors['END']}",end=endl,flush=True)

def read_from_file(file,matrix,docwords,lines=None):
    line_n = 0
    next_percent = .02
    top_str = ''.join(['=' for _ in range(49)])
    top_str = f"{top_str[:20]}PERCENT{top_str[27:]}"
    printc(f"\t[{top_str}]","CYAN")
    printc("\t[","CYAN",endl='')
    for line in file:
        doc, word, count = line.split(' ')
        matrix[int(doc),int(word)] = int(count)
        try:
            docwords[int(doc)].append(count)
        except KeyError:
            docwords[int(doc)] = [int(count)]
        if not lines is None and (float(line_n) / lines) > next_percent:
            printc("=","CYAN",endl='')
            next_percent += .02
        line_n += 1
    printc("]\n","CYAN")

def load_data(vocab_file="vocab.nytimes.txt",dataset_file='docword.nytimes.txt',read_npz=False,npz_name="preSVD.npz",saving_npz=False):
    printc(f"Starting: Data Read","BLUE")
    times = {'start':time()}
    vocab = {}
    docwords = {}


    # READ THE VOCABULARY
    printc(f"\treading vocab from '{Color.colors['RED']}{vocab_file}{Color.colors['TAN']}'","TAN")
    with open(vocab_file,'r') as file:
        for i, word in enumerate(file.readlines()):
            vocab[i] = word
    printc(f"\tread {len(vocab)} words in {(time()-times['start']):.3f} seconds\n","TAN")


    # READING FROM preSVD.npz
    if read_npz:
        # READ THE FULL DATASET
        times['read'] = time()
        fname = f"'{Color.colors['RED']}{npz_name}{Color.colors['TAN']}'"
        printc(f"\treading data from {fname} - precomputed","TAN")
        a = load_npz(str(npz_name))
        printc(f"\tread {fname} in {(time()-times['read']):.3f}\n","TAN")

        printc(f"\tFinished: Data Read in {(time() - times['start']):.3f} seconds","GREEN")
        printc(f"\tReturning: matrix: {a.shape} - size {a.data.size/(1024**2):.2f} MB", "GREEN")
        printc(f"\t           vocab: {len(vocab)}\n\n","GREEN")

        return a, vocab



    # READING DIRECT
    else:
        with open(dataset_file,'r') as file:
            n_articles      = int(file.readline())
            n_words         = int(file.readline())
            n_words_total   = int(file.readline())

            # define the size of the dataset we will build
            rows = n_articles	+	1
            cols = n_words		+	1

            # initialize an lil matrix (faster to fill)
            matrix = lil_matrix( (rows,cols), dtype = np.float64)

            # Step through each article and see which word appeared in it
            times['read'] = time()
            printc(f"\treading data from {dataset_file} - from base file","TAN")

            read_from_file(file,matrix,docwords,lines=n_words_total)
            printc(f"\tread {dataset_file} in {(time()-times['read']):.3f} seconds\n","TAN")

        times['convert'] = time()
        printc(f"\tconverting lil_matrix to csr_matrix","TAN")
        matrix = matrix.tocsr()
        printc(f"\tconverted matrix in {(time()-times['convert']):.3f} seconds\n","TAN")

        if saving_npz:
            times['saving'] = time()
            printc(f"\tsaving matrix to preSVD.npz","TAN")
            save_npz("preSVD.npz",matrix)
            printc(f"\twrote matrix to preSVD.npz in {(time()-times['saving']):.3f}\n","TAN")

        printc(f"\tFinished: Data Read in {(time() - times['start']):.3f} seconds","GREEN")
        printc(f"\tReturning: matrix: {matrix.shape} - size {matrix.data.size/(1024**2):.2f} MB", "GREEN")
        printc(f"\t           vocab: {len(vocab)}\n\n","GREEN")


        return matrix, docwords

def svd_decomp(type='sparse',n=10,matrix=None):
    printc(f"Starting: Data Decomposition","BLUE")

    times = {'start' : time()}
    printc(f"\tbuilding model of type {type}","TAN")
    if type == 'sparse':
        model = TruncatedSVD(n_components=n)
    else:
        model = SVD(n_components=n)
    printc(f"\tmodel built in {(time()-times['start']):.3f} seconds\n","TAN")

    if not input is None:
        times['fit'] = time()
        printc(f"\tfitting matrix shape: {matrix.shape} - to {n} vectors","TAN")
        fitted_data = model.fit_transform(matrix)
        printc(f"\treduced matrix shape: {fitted_data.shape}\n","TAN")
        printc(f"\treduced matrix var  : {model.explained_variance_ratio_.sum():.3f}", "TAN")
        printc(f"\tmatrix reduced in {(time()-times['fit']):.3f} seconds","TAN")
        return fitted_data,model, time()-times['fit']
    else:
        return None, model
