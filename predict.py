import numpy as np
import random
from utils import *
from cv import read_optical_flow_frames


def predict_fire_dummy(n_cams, n_frames):
	#the 1 here indicates that each entry should be the result of 1 coin flip
	return list(np.random.binomial(1, 1/n_cams, n_frames))


#
# given information about the rgb video, create an array of 
# predictions for when to fire
#
# params: camera name (e.g., "cam1"), params for prediction
# returns: array of predictions for when to fire
#
def predict_fire_optical_flow(path, camera, params, ignore_file=False):

	# read rgb
	speeds, sizes, direction = read_optical_flow_frames(path, camera, ignore_file)

	prediction = []
	time_to_wait = round(random() * params["wait time"]) # random start
	for i in range(len(sizes)):
		#print("for " + camera + ", range is " + str(len(sizes)))
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


def predict_fire_ml(n_cams, n_frames):
	
	#
	# To Do
	#

	return list(np.random.binomial(1, 1/n_cams, n_frames))