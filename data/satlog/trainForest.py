#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def main(argv):
	DTrain = np.genfromtxt("sat.trn", delimiter=' ')
	DTest = np.genfromtxt("sat.tst", delimiter=' ')

	# Note from doc: "NB. There are no examples with class 6 in this dataset."
	# Therfore we will map everything from 0-5!

	print(DTrain)
	XTrain = DTrain[:,0:-1].astype(dtype=np.int32)
	YTrain = DTrain[:,-1]
	YTrain = [y-1 if y != 7 else 5 for y in YTrain]
	
	XTest = DTest[:,0:-1].astype(dtype=np.int32)
	YTest = DTest[:,-1]
	YTest = [y-1 if y != 7 else 5 for y in YTest]

	fitModels(True,XTrain,YTrain,XTest,YTest,True)

if __name__ == "__main__":
   main(sys.argv[1:])