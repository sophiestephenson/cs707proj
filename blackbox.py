################################
# blackbox.py
################################

from collections import defaultdict
from config import DIRECTORY
import cv2 as cv
import math
import os
import subprocess
import numpy as np
import sys

from ml import *
from predict import *
from utils import *

#
# runs the simulator using our predictions and returns the values we get
#
# param str scenario: The name of the scenario folder, eg "scenario1"
# param str groundfile: The name of the csv file that contains the ground truths
# param str fire_file: the name of the csv file that contains the firing orders
# returns: a dictionary mapping the camera names to their simulated distances
#
def run_simulator(scenario: str, groundfile:str, fire_file:str):

	ground_path = os.path.join(DATA_DIR, scenario, groundfile)
	fire_path = os.path.join(DATA_DIR, scenario, fire_file)
	outfile_name = fire_file.replace("fire", "output")
	out_path = os.path.join(DATA_DIR, scenario, outfile_name)
	sec_out = "None"
	## run sim using

	if len(sys.argv) > 1:
		# uhhhh i can't get os.system to recognize matlab on my mac so i'm doing this
		sophie_matlab_path = "/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop"
		subprocess.run(sophie_matlab_path + ' -batch "Main2 ' + ground_path + " " + fire_path\
		+ " " + out_path + " " +  sec_out + '"', shell=True)

	else:
		# matlab -batch "Main2 groundfile.csv fire_file.csv output_file.csv sec_out_file.csv"
		os.system('matlab -batch "Main2 ' + ground_path + " " + fire_path\
			+ " " + out_path + " " +  sec_out + '"')

	## get the predicted distances
	return get_matrix(outfile_name)

#
# gets the total diff between the simulated camera distances and the ground truth
# (extended to fit the length of the frames we sent to the simulator)
#
# params: a dictionary mapping the camera names to their simulated distances
# returns: d_hat, the sum of all differences between the ground truth distances
# 			and the simulated distances
#
def compare_to_ground_truth(ground_matrix, simulated_matrix):
	d_hat = 0
	for ground_row, sim_row in zip(ground_matrix, simulated_matrix):
		for i in range(len(ground_row)):
			d_hat += abs(float(ground_row[i]) - float(sim_row[i]))

	return d_hat

#
# given the ground truth file of a particular permutation,
# runs the simulator, calculates the total delta between ground truth and
# estimated distance
#
# params: name of groundfile
# returns: total delta, a float
#
def get_perm_delta(perm_groundfile:str):
	# get the details of the scene
	ground_truth = get_matrix(perm_groundfile)
	n_cams = len(ground_truth)
	n_frames = len(ground_truth[0])

	# dump to sX_pY_fire.csv
	fire_file_name = perm_groundfile.split(".")[0].replace("ground", "fire") + ".csv"
	scenario = "scenario" + fire_file_name.split("_")[0][1:]

	# get the list of fire predictions for each camera
	camnames = ["cam" + str(c + 1) for c in range(n_cams)]

	# the matrix of fire predictions for this permutation
	# this is what gets dumped to the fire csv file
	perm_preds = []

	#the path predict_fire needs to see the .mov files
	path = os.path.join(DATA_DIR, scenario, fire_file_name.replace("fire.csv", ""))
	for cam in camnames:
		cam_path = path + cam
		assert os.path.isdir(cam_path)
		cam_preds = predict_fire_tf(cam_path)
		#cam_preds = predict_fire_dummy(n_cams, n_frames)
		perm_preds.append(cam_preds)

	# dump predictions to outfile
	with open(os.path.join(DATA_DIR, scenario, fire_file_name), "w", newline='') as f:
		csvwriter = csv.writer(f, delimiter=",")
		csvwriter.writerows(perm_preds)

	# returns a matrix of distances, as dumped into outfile by sim
	simulated_distances = run_simulator(scenario, perm_groundfile, fire_file_name)
	return compare_to_ground_truth(ground_truth, simulated_distances)


if __name__ == "__main__":

	#change the working dir to SEC
	if not os.getcwd().endswith("SEC"):
		try:
			os.chdir("SEC")
		except:
			print("No SEC folder?")
			exit(1)

	#get all the datapoints from data folder
	# each scenario folder is named "scenarioX"
	# we sort by the X
	scenarios = [s for s in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, s)) and "scenario" in s]
	scenarios.sort(key=lambda s: int(s[len("scenario"):]))

	# run this loop until we decide not to (?)
	d_hat = math.inf
	while (d_hat > 0):

		#total error
		d_hat = 0
		# scenarioX
		for scenario in scenarios:
			path = os.path.join(DATA_DIR, scenario)
			# step 1: get all the permutations of this scenario
			permutation_grounds = [f for f in os.listdir(path) if f.endswith("ground.csv")]

			#sX_xY_ground.csv
			for perm_groundfile in permutation_grounds:
				print("running " + perm_groundfile)
				d_hat +=  get_perm_delta(perm_groundfile)

		print("d_hat:", d_hat)