#!/usr/bin/env python3

import keras

import numpy as np
from keras.datasets import imdb
from scipy.sparse import csr_matrix

(XTrain, YTrain), (XTest, YTest) = imdb.load_data(num_words=10000,
												  skip_top=30,
												  maxlen=None)
indptr = [0]
indices = []
data = []
for d in XTrain:
	for index in d:
		indices.append(index)
		data.append(len(d))
	indptr.append(len(indices))
XTrain = csr_matrix((data, indices, indptr), dtype=float).toarray()

indptr = [0]
indices = []
data = []
for d in XTest:
	for index in d:
		indices.append(index)
		data.append(len(d))
	indptr.append(len(indices))
XTest = csr_matrix((data, indices, indptr), dtype=float).toarray()

# print("Load IMDB dataset.")
# print(XTrain.shape[0], 'train samples')
# print(XTest.shape[0], 'test samples')
# print(XTrain.shape[1],"features")

#x_train = keras.preprocessing.text.one_hot(x_train,10000)
#print(XTrain[0])

#x_test = keras.preprocessing.text.one_hot(x_test,10000)
#n, D = XTrain.shape    # (n_sample, n_feature)
#d = np.int32(n / 2) * 2 # number of random features
#num_classes = 2

# convert class vectors to binary class matrices
#YTrain = keras.utils.to_categorical(YTrain, num_classes)
#YTest = keras.utils.to_categorical(YTest, num_classes)
# R = np.max(np.linalg.norm(XTrain,2,axis=1))**2
# x_train/=R
# x_test/=R
# R = 1.0

with open("train.csv",'w') as train_file:
	for x,y in zip(XTrain, YTrain):
		line = str(y)
		for xi in x:
			line += "," + str(xi)
		line += "\n"
		train_file.write(line)

with open("test.csv",'w') as test_file:
	for x,y in zip(XTest, YTest):
		line = str(y)
		for xi in x:
			line += "," + str(xi)
		line += "\n"
		test_file.write(line)
