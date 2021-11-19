################################
# blackbox.py
################################

from matplotlib.pyplot import get
from config import DIRECTORY
from cv import optical_flow
from pprint import pprint
from utils import *
import csv
import pickle
import cv
import argparse

# pipeline
# 	1. for each camera:
#   	- read info from video (speed, direction, etc.)
#   	- predict when to fire
#   2. send prediction information to the simulator
#   3. use feedback from the simulator to update predictions


#
# use optical flow to read information about the RGB camera video.
#
# params: camera number, ignore_file (whether to overwrite stored pickles)
# returns: information about the video (speeds, sizes, direction)
#
def read_rbg_frame(camera, ignore_file=False):
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
	direction = smooth_data(get_direction(frame_coords))
	plot(speeds, "rates of change")
	plot(sizes, "sizes")

	pprint(corr_coef(speeds))
	pprint(corr_coef(sizes))
	pprint(direction)

	return (speeds, sizes, direction)

#
# given information about the rgb video, create an array of 
# predictions for when to fire
#
# params: camera name
# returns: array of predictions for when to fire
#
def predict_fire(camera):

	speeds, sizes, directions = read_rbg_frame

	# heuristics:
	# 	- fire more often if object is larger
	# 	- fire more often if object is moving fast 
	# 	- fire more often if object is moving towards you and is bigger

	return []


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', type=str, help='Camera name', default='cam1')
	parser.add_argument('-if', action='store_true', help='Ignore the stored coords file (opt)')
	args = parser.parse_args()

	read_rbg_frame(args["c"], args["if"])


main()