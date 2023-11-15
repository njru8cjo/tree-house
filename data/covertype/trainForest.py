#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def readFile(path):
	f = open(path, 'r')
	header = next(f)
	X = []
	Y = []

	for row in f:
		entries = row.replace("\n","").split(",")

		X.append([float(e) for e in entries[:-1]])
		Y.append(int(entries[-1]))

	# NOTE: It seems, that SKLEarn produces an internal mapping from 0-(|Y| - 1) for classification
	# 		For some reason I was not able to extract this mapping from SKLearn ?!?!
	Y = np.array(Y)
	X = np.array(X)
	Y = Y-min(Y)
	return np.array(X).astype(dtype=np.int32), np.array(Y)

def main(argv):
	X,Y = readFile("covtype.data")

	fitModels(True,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])