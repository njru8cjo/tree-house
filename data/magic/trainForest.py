#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def readFile(path):
	f = open(path, 'r')
	X = []
	Y = []
	for row in f:
		entries = row.strip().split(",")
		x = [float(e) for e in entries[0:-1]]
		y = 0 if entries[-1] == 'g' else 1
		X.append(x)
		Y.append(y)

	return np.array(X).astype(dtype=np.int32), np.array(Y)


def main(argv):
	X,Y = readFile("magic04.data")

	fitModels(True,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])