#!/usr/bin/env python3

import idx2numpy
import numpy as np
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("mnistDirectory")
args=parser.parse_args()

pics=idx2numpy.convert_from_file(args.mnistDirectory+'/train-images-idx3-ubyte')
labels=idx2numpy.convert_from_file(args.mnistDirectory+'/train-labels-idx1-ubyte')
#pics and labels now contains the images and their numerical labels
