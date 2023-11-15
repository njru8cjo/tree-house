#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def main(argv):
	red = np.genfromtxt("winequality-red.csv", delimiter=';', skip_header=1)
	white = np.genfromtxt("winequality-white.csv", delimiter=';', skip_header=1)
	X = np.vstack((red[:,:-1],white[:,:-1])).astype(dtype=np.float32)
	Y = np.concatenate((red[:,-1], white[:,-1]))
	# NOTE: It seems, that SKLEarn produces an internal mapping from 0-(|Y| - 1) for classification
	# 		For some reason I was not able to extract this mapping from SKLearn ?!?!
	Y = Y-min(Y)

	fitModels(False,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])
