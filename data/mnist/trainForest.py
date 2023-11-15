#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels
def readFile(path):
	X = []
	Y = []

	f = open(path,'r')

	for row in f:
		entries = row.strip("\n").split(",")
		
		Y.append(int(entries[0])-1)
		x = [int(e) for e in entries[1:]]
		X.append(x)

	Y = np.array(Y)-min(Y)
	return np.array(X).astype(dtype=np.int32), Y

def main(argv):
	XTrain,YTrain = readFile("train.csv")
	XTest,YTest = readFile("test.csv")

	fitModels(True,XTrain,YTrain,XTest,YTest)

if __name__ == "__main__":
   main(sys.argv[1:])