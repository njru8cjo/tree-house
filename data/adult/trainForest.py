#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def getFeatureVector(entries):
	x = []
	x.append(float(entries[0])) # age = continous

	workclass = [0 for i in range(8)]
	if entries[1] == "Private":
		workclass[0] = 1
	elif entries[1] == "Self-emp-not-inc":
		workclass[1] = 1
	elif entries[1] == "Self-emp-inc":
		workclass[2] = 1
	elif entries[1] == "Federal-gov":
		workclass[3] = 1
	elif entries[1] == "Local-gov":
		workclass[4] = 1
	elif entries[1] == "State-gov":
		workclass[5] = 1
	elif entries[1] == "Without-pay":
		workclass[6] = 1
	else: #Never-worked
		workclass[7] = 1
	x.extend(workclass)
	x.append(float(entries[2]))

	education = [0 for i in range(16)]
	if entries[3] == "Bachelors":
		education[0] = 1
	elif entries[3] == "Some-college":
		education[1] = 1
	elif entries[3] == "11th":
		education[2] = 1
	elif entries[3] == "HS-grad":
		education[3] = 1
	elif entries[3] == "Prof-school":
		education[4] = 1
	elif entries[3] == "Assoc-acdm":
		education[5] = 1
	elif entries[3] == "Assoc-voc":
		education[6] = 1
	elif entries[3] == "9th":
		education[7] = 1
	elif entries[3] == "7th-8th":
		education[8] = 1
	elif entries[3] == "12th":
		education[9] = 1
	elif entries[3] == "Masters":
		education[10] = 1
	elif entries[3] == "1st-4th":
		education[11] = 1
	elif entries[3] == "10th":
		education[12] = 1
	elif entries[3] == "Doctorate":
		education[13] = 1
	elif entries[3] == "5th-6th":
		education[14] = 1
	if entries[3] == "Preschool":
		education[15] = 1

	x.extend(education)
	x.append(float(entries[4]))

	marital = [0 for i in range(7)]
	if entries[5] == "Married-civ-spouse":
		marital[0] = 1
	elif entries[5] == "Divorced":
		marital[1] = 1
	elif entries[5] == "Never-married":
		marital[2] = 1
	elif entries[5] == "Separated":
		marital[3] = 1
	elif entries[5] == "Widowed":
		marital[4] = 1
	elif entries[5] == "Married-spouse-absent":
		marital[5] = 1
	else:
		marital[6] = 1
	x.extend(marital)

	occupation = [0 for i in range(14)]
	if entries[6] == "Tech-support":
		occupation[0] = 1
	elif entries[6] == "Craft-repair":
		occupation[1] = 1
	elif entries[6] == "Other-service":
		occupation[2] = 1
	elif entries[6] == "Sales":
		occupation[3] = 1
	elif entries[6] == "Exec-managerial":
		occupation[4] = 1
	elif entries[6] == "Prof-specialty":
		occupation[5] = 1
	elif entries[6] == "Handlers-cleaners":
		occupation[6] = 1
	elif entries[6] == "Machine-op-inspct":
		occupation[7] = 1
	elif entries[6] == "Adm-clerical":
		occupation[8] = 1
	elif entries[6] == "Farming-fishing":
		occupation[9] = 1
	elif entries[6] == "Transport-moving":
		occupation[10] = 1
	elif entries[6] == "Priv-house-serv":
		occupation[11] = 1
	elif entries[6] == "Protective-serv":
		occupation[12] = 1
	else:
		occupation[13] = 1
	x.extend(occupation)

	relationship = [0 for i in range(6)]
	if entries[7] == "Wife":
		relationship[0] = 1
	elif entries[7] == "Own-child":
		relationship[1] = 1
	elif entries[7] == "Husband":
		relationship[2] = 1
	elif entries[7] == "Not-in-family":
		relationship[3] = 1
	elif entries[7] == "Other-relative":
		relationship[4] = 1
	else:
		relationship[5] = 1
	x.extend(relationship)

	race = [0 for i in range(5)]
	if entries[8] == "White":
		race[0] = 1
	elif entries[8] == "Asian-Pac-Islander":
		race[1] = 1
	elif entries[8] == "Amer-Indian-Eskimo":
		race[2] = 1
	elif entries[8] == "Other":
		race[3] = 1
	else:
		race[4] = 1
	x.extend(race)

	if (entries[9] == "Male"):
		x.extend([1,0])
	else:
		x.extend([0,1])

	x.append(float(entries[10]))
	x.append(float(entries[11]))
	x.append(float(entries[12]))

	native = [0 for i in range(42)]
	if entries[14] == "United-States":
		native[1] = 1
	elif entries[14] == "Cambodia":
		native[2] = 1
	elif entries[14] == "England":
		native[3] = 1
	elif entries[14] == "Puerto-Rico":
		native[4] = 1
	elif entries[14] == "Canada":
		native[5] = 1
	elif entries[14] == "Germany":
		native[6] = 1
	elif entries[14] == "Outlying-US(Guam-USVI-etc)":
		native[7] = 1
	elif entries[14] == "India":
		native[8] = 1
	elif entries[14] == "Japan":
		native[9] = 1
	elif entries[14] == "Greece":
		native[10] = 1
	elif entries[14] == "South":
		native[11] = 1
	elif entries[14] == "China":
		native[12] = 1
	elif entries[14] == "Cuba":
		native[13] = 1
	elif entries[14] == "Iran":
		native[14] = 1
	elif entries[14] == "Honduras":
		native[15] = 1
	elif entries[14] == "Philippines":
		native[16] = 1
	elif entries[14] == "Italy":
		native[17] = 1
	elif entries[14] == "Poland":
		native[18] = 1
	elif entries[14] == "Jamaica":
		native[19] = 1
	elif entries[14] == "Vietnam":
		native[20] = 1
	elif entries[14] == "Mexico":
		native[21] = 1
	elif entries[14] == "Portugal":
		native[22] = 1
	elif entries[14] == "Ireland":
		native[23] = 1
	elif entries[14] == "France":
		native[24] = 1
	elif entries[14] == "Dominican-Republic":
		native[25] = 1
	elif entries[14] == "Laos":
		native[26] = 1
	elif entries[14] == "Ecuador":
		native[27] = 1
	elif entries[14] == "Taiwan":
		native[28] = 1
	elif entries[14] == "Haiti":
		native[29] = 1
	elif entries[14] == "Columbia":
		native[30] = 1
	elif entries[14] == "Hungary":
		native[31] = 1
	elif entries[14] == "Guatemala":
		native[32] = 1
	elif entries[14] == "Nicaragua":
		native[33] = 1
	elif entries[14] == "Scotland":
		native[34] = 1
	elif entries[14] == "Thailand":
		native[35] = 1
	elif entries[14] == "Yugoslavia":
		native[36] = 1
	elif entries[14] == "El-Salvador":
		native[37] = 1
	elif entries[14] == "Trinadad&Tobago":
		native[38] = 1
	elif entries[14] == "Peru":
		native[39] = 1
	elif entries[14] == "Hong":
		native[40] = 1
	else:
		native[41] = 1

	return x

def main(argv):
	X = []
	Y = []

	f = open("adult.data")
	for row in f:
		if (len(row) > 1):
			entries = row.replace("\n", "").replace(" ", "").split(",")

			x = getFeatureVector(entries)

			if (entries[-1] == "<=50K"):
				y = 0
			else:
				y = 1

			Y.append(y)
			X.append(x)

	X = np.array(X).astype(dtype=np.int32)
	Y = np.array(Y)

	fitModels(True,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])
