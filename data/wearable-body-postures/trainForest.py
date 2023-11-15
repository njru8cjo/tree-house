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
		entries = row.replace("\n","").split(";")

		if entries[-1] == 'sitting':
			y = 0
		elif entries[-1] == 'standing':
			y = 1
		elif entries[-1] == 'standingup':
			y = 2
		elif entries[-1] == 'walking':
			y = 3
		elif entries[-1] == 'sittingdown':
			y = 4
		else:
			print("ERROR READING CLASSES:", entries[-1])
		
		x = []
		if entries[1] == 'Man':
			x.append(0)
		else:
			x.append(1)
			
		for e in entries[2:-1]:
			x.append(int(float(e.replace(",","."))*100))
		X.append(x)
		Y.append(y)

	return np.array(X).astype(dtype=np.int32), np.array(Y)

def main(argv):
	X,Y = readFile("dataset-har-PUC-Rio-ugulino.csv")

	fitModels(True,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])