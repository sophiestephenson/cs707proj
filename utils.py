################################
# utils.py
################################

import math
import numpy as np
from shapely import geometry
import os
import matplotlib.pyplot as plt
from config import *
import csv
import statistics

#################################
# GETTING DATA FROM FRAME COORDS
#################################

#
# uses optical flow CV to identify the change in position per frame and therefore
# the speed of movement between different frames.
#
# params: frame_coords (an array of frames, each frame has a set of coordinates 
#  						corresponding to the different tracked points)
# returns: an array of differences between the frames
#
def get_speeds(frame_coords):

	speeds = []
	for i in range(1, len(frame_coords)):

		old_frame = frame_coords[i - 1]
		new_frame = frame_coords[i]

		diffs = []
		for j in range(len(old_frame)):
			diff = coord_change(old_frame[j], new_frame[j])
			diffs.append(diff)

		speeds.append(np.mean(diffs))

	return speeds

#
# for each frame, approximates the size of the shape at that time by
# using the area of the shape formed by the coordinates.
#
# params: frame (an array of frames, each frame has a set of coordinates 
#  						corresponding to the different tracked points)
# returns: an array representing the approximate size of the object at each frame
#
def get_sizes(frame_coords):

	sizes = []
	for frame in frame_coords:
		s = approx_size(frame)
		sizes.append(s)

	return sizes

#
# for each pair of frames, identifies the general direction of movement 
# for use in the black box. (very rough approximation)
#
# params: frame (an array of frames, each frame has a set of coordinates 
#  						corresponding to the different tracked points)
# returns: the direction of the object
#
def get_direction(frame_coords):

	directions = []
	for i in range(1, len(frame_coords)):

		old_frame = frame_coords[i - 1]
		new_frame = frame_coords[i]
		dir = obj_direction(old_frame, new_frame)
		directions.append(dir)
		
	# combine the info
	mode = statistics.mode(directions)
	return mode


#################################
# HELPERS
#################################

#
# given two coordinates (x, y), calculate the distance between them
#
# params: old coords, new coords
# returns: the distance between them: sqrt((x1 - x2)^2 + (y1 - y2)^2)
#
def coord_change(old_coords, new_coords):
	x1, y1 = old_coords
	x2, y2 = new_coords
	return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

#
# given a set of coordinates (x, y), find the area of the polygon they make
# (meant as a way to approximate the size of the object represented by the coords)
#
# params: a set of coords (x, y)
# returns: the area of the polygon formed by the coords
#
def approx_size(coords_set):
	tups = []
	for c in coords_set:
		x, y = c
		tups.append((x, y))

	try:
		polygon = geometry.Polygon(tups)
		return polygon.area
	except ValueError:
		# can't get the size for some reason, assume it's because there are <= 2 coords. 
		if len(coords_set) >= 2:
			return coord_change(coords_set[0], coords_set[1])
		else:
			return 1

#
# given two sets of coords, returns the (rough) direction of the object
#
# params: old_coords_set, new_coords_set
# returns: the approx direction of the object (left, right, away, towards)
#
def obj_direction(old_coords_set, new_coords_set):

	xdirs = []
	ydirs = []
	for j in range(len(old_coords_set)):
		x1, y1 = old_coords_set[j]
		x2, y2 = new_coords_set[j]
		xdirs.append(x1 - x2)
		ydirs.append(y1 - y2)

	if (all(map(lambda x: x > 0, xdirs))): return "left"
	if (all(map(lambda x: x < 0, xdirs))): return "right"

	# compare the first two coords for away and toward (rough)
	old_x1, y1 = old_coords_set[0]
	old_x2, y2 = old_coords_set[1]
	new_x1, y3 = new_coords_set[0]
	new_x2, y4 = new_coords_set[1]
	if (old_x1 - old_x2) > (new_x1 - new_x2): return "away"
	elif (old_x1 - old_x2) < (new_x1 - new_x2): return "towards"
	else: return "not sure"


#
# smooths data to remove noise
#
# params: list of data
# returns: smoothed list
#
# https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
def smooth_data(l):
	kernel = np.ones(SMOOTHING_KERNEL_SIZE) / SMOOTHING_KERNEL_SIZE
	return np.convolve(l, kernel, mode="same")


#
# rejects outliers using the IQR
# doesn't look great
#
# params: list of data
# returns: data without outliers (outliers become the previous value)
#
# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list 
def reject_outliers(data):
	Q1, Q3 = np.quantile(data, [0.25, 0.75])
	IQR = Q3 - Q1

	def isOutlier(x):
		return (x < (Q1 - 1.5 * IQR) or x > (Q3 + 1.5 * IQR))

	outliers = list(map(isOutlier, data))
	
	for i in range(len(data)):
		if outliers[i] and i > 0:
			print("removed outlier:", data[i])
			data[i] = data[i - 1]
	return data

#
# create a plot of the data with the given title
# 
# params: data, title
# returns: nothing, but prints a plot
#
def plot(data, title):
	plt.plot(data)
	plt.title(title)
	plt.show()


#
# get the correlation coefficient of the data
#
# params: list of data
# returns: the correlation coefficient
#
def corr_coef(data):
	return np.corrcoef(data, np.arange(len(data)))[0][1]


# dumbly get the matrix from file
# The file can be either the ground truth or the simulated distances
# each row corresponds to a camera, each column corresponds to a frame
# params: name of groundfile: sX_pY_ground.csv
# returns: matrix of ground truths. rows are cameras, columns are frames
def get_matrix(file: str):
	scenario = "scenario" + file.split("_")[0][1:]
	maxtrix = []
	with open(os.path.join(DATA_DIR, scenario, file), 'r') as f:
		reader = csv.reader(f, delimiter=",")
		matrix = list(reader)

	#normalize the ground file. Processing produces big numbers
	#the sim doesn't like big numbers
	if "ground" in file:
		for r in range(len(matrix)):
			for c in range(len(matrix[r])):
				matrix[r][c] = float(matrix[r][c])/100

	return matrix


# DEPRECATED:
# get the ground truth from saed file and map it to a specific length list
#
# params: camera number, the size of the list to create
# returns: the ground truth, mapped to the appropriate size list
# def get_ground_truth(camera, size):
#
# 	# grab gt from file
# 	ground_truth = []
# 	with open(DIRECTORY + "row_per_cam.csv", 'r') as f:
# 		reader = csv.reader(f)
# 		for row in reader:
# 			if row[0] == camera:
# 				ground_truth = row[1:]
# 				break
# 	orig_len = len(ground_truth)
#
# 	# map the gp to a new list
# 	ground_truth_new_size = []
# 	for i in range(size):
# 		gt_i = round((i/size) * orig_len)
# 		ground_truth_new_size.append(float(ground_truth[gt_i]))
#	return ground_truth_new_size

def get_number_of_cameras(groundfile: str):
	with open(groundfile) as f:
		return len(f.readlines())

#
# literally just returns infinit (for use in creating default dicts)
#
def inf():
	return math.inf
