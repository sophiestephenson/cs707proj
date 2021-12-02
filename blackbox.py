################################
# blackbox.py
################################

from collections import defaultdict
from config import DIRECTORY
from cv import optical_flow
from pprint import pprint
from utils import *
import pickle
import cv2 as cv
from random import random
import math
import os

#
# use optical flow to read information about the RGB camera video.
#
# params: camera number, ignore_file (whether to overwrite stored pickles)
# returns: information about the video (speeds, sizes, direction)
#
def read_rbg_frames(camera, ignore_file=False):
	vid_filename = DIRECTORY + camera + ".mov"
	pickle_filename = DIRECTORY + camera + "_coords.pkl"

	# get saved frame data
	try:
		if ignore_file:
			raise FileNotFoundError
		frame_coords = pickle.load(open(pickle_filename, "rb"))

	# if no saved frame data, generate frame data and store it for next time
	except FileNotFoundError:
		f = open(pickle_filename, "wb")
		capture = cv.VideoCapture(cv.samples.findFileOrKeep(vid_filename))
		if not capture.isOpened():
			print("Unable to open", vid_filename)
			exit(0)
		frame_coords = optical_flow(capture)
		pickle.dump(frame_coords, f)
		f.close()

	# get data about the rbg frame
	speeds = smooth_data(get_speeds(frame_coords))
	sizes = smooth_data(get_sizes(frame_coords))
	direction = get_direction(frame_coords)

	# print information, but only if we are ignoring the file
	if (ignore_file):
		plot(speeds, "rates of change")
		plot(sizes, "sizes")
		#print("speed r:", corr_coef(speeds))
		#print("size r:", corr_coef(sizes))
		pprint(direction)

	#return (speeds, sizes, direction)
	#simplifying this for now
	return frame_coords


#
# given information about the rgb video, create an array of 
# predictions for when to fire
#
# params: camera name (e.g., "cam1"), params for prediction
# returns: array of predictions for when to fire
#
def predict_fire(camera, params, ignore_file=False):

	# read rgb
	speeds, sizes, direction = read_rbg_frames(camera, ignore_file)

	prediction = []
	time_to_wait = round(random() * params["wait time"]) # random start
	for i in range(len(sizes)):
		if time_to_wait > 0:
			time_to_wait -= 1
			prediction.append(0)
			continue

		probability = 1
		if params["size factor"] != 0:
			probability *= sizes[i] * 1./params["size factor"]
		if params["speed factor"] != 0 and i < len(speeds):
			probability *= speeds[i] * 1./params["speed factor"]
		if params["direction factor"] != 0 and direction == 'away':
			probability -= 1./(params["direction factor"] * 100)
		if params["direction factor"] != 0 and direction == 'towards':
			probability += 1./(params["direction factor"] * 100)
		
		if probability > 1: 
			probability = 1
			time_to_wait = params["wait time"]
		if probability < 0: 
			probability = 0

		prediction.append(round(probability))

	if ignore_file: plot(prediction, "predicted firing")

	return prediction


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
	# matlab -batch "Main2 groundfile.csv fire_file.csv output_file.csv sec_out_file.csv"
	os.system('matlab -batch "Main2 ' + ground_path + " " + fire_path\
		+ " " + out_path + " " +  sec_out + '"')
	## get the predicted distances
	return get_matrix(out_path)

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
			d_hat += abs(ground_row[i] - sim_row[i])

	return d_hat

# given the ground truth file of a particular permutation,
# runs the simulator, calculates the total delta between ground truth and
# estimated distance
# params: name of groundfile
# returns: total delta, a float
def get_perm_delta(perm_groundfile:str, ignore_file):
	ground_truth = get_matrix(perm_groundfile)
	n_cams = len(ground_truth)

	# get the list of fire predictions for each camera
	camnames = ["cam" + str(c + 1) for c in range(n_cams)]

	# the matrix of fire predictions for this permutation
	# this is what gets dumped to the fire csv file
	perm_preds = []
	for cam in camnames:
		cam_preds = predict_fire(cam, params, ignore_file)
		perm_preds.append(cam_preds)

	# dump to sX_pY_fire.csv
	fire_file_name = perm_groundfile.split(".")[0].replace("ground",
														   "fire") + ".csv"
	scenario = "scenario" + fire_file_name.split("_")[0][1:]
	with open(os.path.join(DATA_DIR, scenario, fire_file_name), "w") as f:
		csvwriter = csv.writer(csvfile=f, delimiter=",")
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
	scenarios = []
	for scenario in os.listdir("data"):
		# scenario is something like "scenario1
		scenarios.append(os.path.join("data", scenario))
	# each scenario folder is named "scenarioX"
	# we sort by the X
	scenarios.sort(key=lambda s: int(s[len("scenario"):]))

	# pipeline
	#	1. for each camera:
	#   	- read info from video (rate of change (speed), size, and direction)
	#   	- predict when to fire based on this info
	#   2. send prediction information to the simulator and run
	#   3. use feedback from the simulator to update predictions
	#	4. select new parameters and run again
	#

	## initialize the factor dictionaries
	wait_times 			= defaultdict(inf)
	size_factors 		= defaultdict(inf)
	speed_factors 		= defaultdict(inf)
	direction_factors 	= defaultdict(inf)

	# epsilon value for exploration (higher = more exploration)
	epsilon = 0.5

	# set our starter parameters that we envision will work well
	# heuristics:
	# 	- fire more often if object is larger
	# 	- fire more often if object is moving fast 
	# 	- fire more often if object is moving towards you and is bigger
	#   - only fire once every 5 frames
	params = {
		"wait time": 5,
		"size factor": 500,
		"speed factor": 2,
		"direction factor": 1,
	}

	# run this loop until we decide not to (?)
	d_hat = math.inf
	ignore_file = True
	while (d_hat > 0):

		#total error
		d_hat = 0
		# scenarioX
		for scenario in scenarios:
			# step 1: get all the permutations of this scenario
			permutation_grounds = [f for f in os.listdir(scenario) if f.endswith("ground.csv")]

			#sX_xY_ground.csv
			for perm_groundfile in permutation_grounds:
				total_delta = get_perm_delta(perm_groundfile, ignore_file)
				d_hat += total_delta
				#only ignore file on the first run
				ignore_file = False

				# show that d_hat is added to with each run of the sim
				print("d_hat:", d_hat)


		# update factor dicts with d_hat (if d_hat is lower than the existing)
		size_factors[params["size factor"]] 			= min(d_hat, size_factors[params["size factor"]])
		speed_factors[params["speed factor"]] 			= min(d_hat, speed_factors[params["speed factor"]] )
		direction_factors[params["direction factor"]]	= min(d_hat, direction_factors[params["direction factor"]])
		wait_times[params["wait time"]] 				= min(d_hat, wait_times[params["wait time"]])

		for d in (size_factors, speed_factors, direction_factors, wait_times):
			pprint(d)
			print("\n")

		# take greedy action with probability 1 - epsilon
		# (pick the best performing value so far for each factor)
		if random() > epsilon:
			params["wait time"] 		= min(wait_times, key = wait_times.get)
			params["size factor"] 		= min(size_factors, key = size_factors.get)
			params["speed factor"] 		= min(speed_factors, key = speed_factors.get)
			params["direction factor"] 	= min(direction_factors, key = direction_factors.get)

			## TO DO: do something if we choose the exact same params as last time

		# otherwise, explore (pick a random value for each)
		# wait times go from 0 to WAIT_TIME_MAX; others go from -range/2 to +range/2
		else:
			params["wait time"] 		= int(random() * WAIT_TIME_MAX)
			params["size factor"] 		= int((random() * SIZE_FACTOR_RANGE) - SIZE_FACTOR_RANGE/2)
			params["speed factor"] 		= int((random() * SPEED_FACTOR_RANGE) - SPEED_FACTOR_RANGE/2)
			params["direction factor"] 	= int((random() * DIRECTION_FACTOR_RANGE) - DIRECTION_FACTOR_RANGE/2)