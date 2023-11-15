#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def getFeatureVector(entries):
	x = []
	x.append(float(entries[0])) # age = continous

	job = [0 for i in range(12)]
	if entries[1] == "admin.":
		job[0] = 1
	elif entries[1] == "blue-collar":
		job[1] = 1
	elif entries[1] == "entrepreneur":
		job[2] = 1
	elif entries[1] == "housemaid":
		job[3] = 1
	elif entries[1] == "management":
		job[4] = 1
	elif entries[1] == "retired":
		job[5] = 1
	elif entries[1] == "self-employed":
		job[6] = 1
	elif entries[1] == "services":
		job[7] = 1
	elif entries[1] == "student":
		job[8] = 1
	elif entries[1] == "technician":
		job[9] = 1
	elif entries[1] == "unemployed":
		job[10] = 1
	else:
		job[11] = 1

	x.extend(job)

	martial = [0 for i in range(4)]
	if entries[2] == "divorced":
		martial[0] = 1
	elif entries[2] == "married":
		martial[1] = 1
	elif entries[2] == "single":
		martial[2] = 1
	else:
		martial[3] = 1

	education = [0 for i in range(8)]
	if entries[3] == "basic.4y":
		education[0] = 1
	elif entries[3] == "basic.6y":
		education[1] = 1
	elif entries[3] == "basic.9y":
		education[2] = 1
	elif entries[3] == "high.school":
		education[3] = 1
	elif entries[3] == "illiterate":
		education[4] = 1
	elif entries[3] == "professional.course":
		education[5] = 1
	elif entries[3] == "university.degree":
		education[6] = 1
	else:
		education[7] = 1
	x.extend(education)


	if entries[4] == "no":
		x.extend([1,0,0])
	elif entries[4] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[5] == "no":
		x.extend([1,0,0])
	elif entries[5] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[6] == "no":
		x.extend([1,0,0])
	elif entries[6] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[7] == "telephone":
		x.append(0)
	else:
		x.append(1)

	month = [0 for i in range(12)]
	if entries[8] == "jan":
		month[0] = 1
	elif entries[8] == "feb":
		month[1] = 1
	elif entries[8] == "mar":
		month[2] = 1
	elif entries[8] == "apr":
		month[3] = 1
	elif entries[8] == "may":
		month[4] = 1
	elif entries[8] == "jun":
		month[5] = 1
	elif entries[8] == "jul":
		month[6] = 1
	elif entries[8] == "aug":
		month[7] = 1
	elif entries[8] == "sep":
		month[8] = 1
	elif entries[8] == "oct":
		month[9] = 1
	elif entries[8] == "nov":
		month[10] = 1
	else:
		month[11] = 1
	x.extend(month)

	day = [0 for i in range(5)]
	if entries[9] == "mon":
		day[0] = 1
	elif entries[9] == "tue":
		day[1] = 1
	elif entries[9] == "wed":
		day[2] = 1
	elif entries[9] == "thu":
		day[3] = 1
	else:
		day[4] = 1
	x.extend(day)

	#x.append(float(entries[10]))
	x.append(int(float(entries[11])*1000))
	x.append(int(float(entries[12])*1000))
	x.append(int(float(entries[13])*1000))

	if entries[14] == "failure":
		x.extend([1,0,0])
	elif entries[14] == "nonexistent":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	x.append(int(float(entries[15])*1000))
	x.append(int(float(entries[16])*1000))
	x.append(int(float(entries[17])*1000))
	x.append(int(float(entries[18])*1000))
	x.append(int(float(entries[19])*1000))

	return x

def main(argv):
	X = []
	Y = []

	f = open("bank-additional/bank-additional-full.csv")
	next(f)
	for row in f:
		if (len(row) > 1):
			entries = row.replace("\n", "").replace(" ", "").replace("\"","").split(";")

			x = getFeatureVector(entries)

			if (entries[-1] == "no"):
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
