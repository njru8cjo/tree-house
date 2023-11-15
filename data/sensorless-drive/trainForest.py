#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def main(argv):
	data = np.genfromtxt("Sensorless_drive_diagnosis.txt", delimiter=' ')
	idx = np.random.permutation(len(data))

	X = data[idx,0:-1].astype(dtype=np.float32)
	Y = data[idx,-1]
	Y = Y-min(Y)

	fitModels(False,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])
