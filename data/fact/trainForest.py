#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels

def readFile():
	header= ("conc_core","concentration_one_pixel","concentration_two_pixel","leakage2","size","width","length","conc_cog","m3l","m3t","num_islands","num_pixel_in_shower","ph_charge_shower_max","ph_charge_shower_mean","ph_charge_shower_min","ph_charge_shower_variance")
	#header = ("arr_time_pedestal_kurtosis","arr_time_pedestal_max","arr_time_pedestal_mean","arr_time_pedestal_median","arr_time_pedestal_min","arr_time_pedestal_p25","arr_time_pedestal_p75","arr_time_pedestal_skewness","arr_time_pedestal_variance","arr_time_pos_shower_kurtosis","arr_time_pos_shower_max","arr_time_pos_shower_mean","arr_time_pos_shower_min","arr_time_pos_shower_skewness","arr_time_pos_shower_variance","arr_time_shower_kurtosis","arr_time_shower_max","arr_time_shower_mean","arr_time_shower_min","arr_time_shower_skewness","arr_time_shower_variance","arrival_time_mean","cog_x","cog_y","conc_cog","conc_core","concentration_one_pixel","concentration_two_pixel","delta","fluct_mean_kurtosis","fluct_mean_max","fluct_mean_mean","fluct_mean_median","fluct_mean_min","fluct_mean_p25","fluct_mean_p75","fluct_mean_skewness","fluct_mean_variance","fluct_median_kurtosis","fluct_median_max","fluct_median_mean","fluct_median_median","fluct_median_min","fluct_median_p25","fluct_median_p75","fluct_median_skewness","fluct_median_variance","fluct_std_kurtosis","fluct_std_max","fluct_std_mean","fluct_std_median","fluct_std_min","fluct_std_p25","fluct_std_p75","fluct_std_skewness","fluct_std_variance","fluct_sum_kurtosis","fluct_sum_max","fluct_sum_mean","fluct_sum_median","fluct_sum_min","fluct_sum_p25","fluct_sum_p75","fluct_sum_skewness","fluct_sum_variance","fluct_var_kurtosis","fluct_var_max","fluct_var_mean","fluct_var_median","fluct_var_min","fluct_var_p25","fluct_var_p75","fluct_var_skewness","fluct_var_variance","leakage","leakage2","length","m3_long","m3_trans","m3l","m3t","m4_long","m4_trans","max_pos_pedestal_kurtosis","max_pos_pedestal_max","max_pos_pedestal_mean","max_pos_pedestal_median","max_pos_pedestal_min","max_pos_pedestal_p25","max_pos_pedestal_p75","max_pos_pedestal_skewness","max_pos_pedestal_variance","max_pos_shower_kurtosis","max_pos_shower_max","max_pos_shower_mean","max_pos_shower_min","max_pos_shower_skewness","max_pos_shower_variance","max_slopes_pedestal_kurtosis","max_slopes_pedestal_max","max_slopes_pedestal_mean","max_slopes_pedestal_median","max_slopes_pedestal_min","max_slopes_pedestal_p25","max_slopes_pedestal_p75","max_slopes_pedestal_skewness","max_slopes_pedestal_variance","max_slopes_pos_shower_kurtosis","max_slopes_pos_shower_max","max_slopes_pos_shower_mean","max_slopes_pos_shower_min","max_slopes_pos_shower_skewness","max_slopes_pos_shower_variance","max_slopes_shower_kurtosis","max_slopes_shower_max","max_slopes_shower_mean","max_slopes_shower_min","max_slopes_shower_skewness","max_slopes_shower_variance","num_islands","num_pixel_in_pedestal","num_pixel_in_shower","ped_mean_kurtosis","ped_mean_max","ped_mean_mean","ped_mean_median","ped_mean_min","ped_mean_p25","ped_mean_p75","ped_mean_skewness","ped_mean_variance","ped_median_kurtosis","ped_median_max","ped_median_mean","ped_median_median","ped_median_min","ped_median_p25","ped_median_p75","ped_median_skewness","ped_median_variance","ped_std_kurtosis","ped_std_max","ped_std_mean","ped_std_median","ped_std_min","ped_std_p25","ped_std_p75","ped_std_skewness","ped_std_variance","ped_sum_kurtosis","ped_sum_max","ped_sum_mean","ped_sum_median","ped_sum_min","ped_sum_p25","ped_sum_p75","ped_sum_skewness","ped_sum_variance","ped_var_kurtosis","ped_var_max","ped_var_mean","ped_var_median","ped_var_min","ped_var_p25","ped_var_p75","ped_var_skewness","ped_var_variance","pedestal_size","pedestal_timespread","ph_charge_pedestal_kurtosis","ph_charge_pedestal_max","ph_charge_pedestal_mean","ph_charge_pedestal_median","ph_charge_pedestal_min","ph_charge_pedestal_p25","ph_charge_pedestal_p75","ph_charge_pedestal_skewness","ph_charge_pedestal_variance","ph_charge_shower_kurtosis","ph_charge_shower_max","ph_charge_shower_mean","ph_charge_shower_min","ph_charge_shower_skewness","ph_charge_shower_variance","photoncharge_mean","size","slope_long","slope_spread","slope_spread_weighted","slope_trans","timespread","timespread_weighted","width")
	X = []
	Y = []
	#MAXROWS = 250000

	print("Reading Gamma data")
	GData = np.genfromtxt("gamma_simulations_facttools_dl2.csv", delimiter=',',usecols=header, names = True)#,max_rows=MAXROWS)
	
	dropped = 0
	included = 0
	for x in GData:
		tmp = np.array([float(xi) for xi in x])
		if np.any(np.isnan(tmp)) or np.any(np.isinf(tmp)):
			dropped += 1
		else:
			included += 1
			X.append(tmp)
		
	for i in range(included):
		Y.append(0)

	print("Reading Proton data")
	PData = np.genfromtxt("proton_simulations_facttools_dl2.csv", delimiter=',',usecols=header, names = True)#,max_rows=MAXROWS)
	included = 0
	for x in PData:
		tmp = np.array([float(xi) for xi in x])
		if np.any(np.isnan(tmp)) or np.any(np.isinf(tmp)):
			dropped += 1
		else:
			included += 1
			X.append(tmp)
		
	for i in range(included):
		Y.append(1)

	print("Dropped", dropped, "data points because of NaN")
	print(len(X), "data points still available")

	return np.array(X).astype(dtype=np.float32), np.array(Y)


def main(argv):
	X,Y = readFile()

	fitModels(False,X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])