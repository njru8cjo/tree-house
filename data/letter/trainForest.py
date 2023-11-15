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
		x = [int(e) for e in entries[1:]]
		# Labels are capital letter has. 'A' starts in ASCII code with 65
		# We map it to '0' here, since SKLearn internally starts with mapping = 0 
		# and I have no idea how it produces correct outputs in the first place
		y = ord(entries[0]) - 65
		
		X.append(x)
		Y.append(y)

	return np.array(X).astype(dtype=np.int32), np.array(Y)

def main(argv):
	X,Y = readFile("letter-recognition.data")

	fitModels(True,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])