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
		entries = row.replace("\n","").split(",")
		x = [float(e) for e in entries[1:]]
		y = int(entries[0])

		X.append(x)
		Y.append(y)

	return np.array(X).astype(dtype=np.float32), np.array(Y)

def main(argv):
	XTrain,YTrain = readFile("train.csv")
	XTest,YTest = readFile("test.csv")

	fitModels(False,XTrain,YTrain,XTest,YTest)

if __name__ == "__main__":
   main(sys.argv[1:])
