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

	return (speeds, sizes, direction)


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

		prediction.append(probability)

	plot(prediction, "predicted firing")

	return prediction


#
# runs the simulator using our predictions and returns the values we get
#
# params: a dictionary mapping the camera names to their prediction arrays
# returns: a dictionary mapping the camera names to their simulated distances
#
def run_simulator(cam_predictions):
	
	## TO DO!!!

	## send predictions to simulator
	## run it
	## get the predicted distances
	## return them

	simulated_distances = {}

	for k in cam_predictions.keys():
		gt = get_ground_truth(k, len(cam_predictions[k]))
		simulated_distances[k] = [x + (30 * random()) - 15 for x in gt]

	return simulated_distances


#
# gets the total diff between the simulated camera distances and the ground truth
# (extended to fit the length of the frames we sent to the simulator)
#
# params: a dictionary mapping the camera names to their simulated distances
# returns: d_hat, the sum of all differences between the ground truth distances
# 			and the simulated distances
#
def compare_to_ground_truth(simulated_distances):
	d_hat = 0
	for cam in simulated_distances.keys():
		sim = simulated_distances[cam]
		gt = get_ground_truth(cam, len(sim))
		for i in range(len(gt)):
			d_hat += abs(gt[i] - sim[i])

	return d_hat
	


if __name__ == "__main__":

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

		# get predictions for the two cameras
		cam1_preds = predict_fire("cam1", params, ignore_file)
		cam2_preds = predict_fire("cam2", params, ignore_file)
		ignore_file = False # after the first round, use the saved file

		# run the simulator with these predictions
		simulated_distances = run_simulator({"cam1": cam1_preds, "cam2": cam2_preds})

		# compare results to the ground truth 
		d_hat = compare_to_ground_truth(simulated_distances)
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