################################
# utils.py
################################

import math
#import pickle
import tensorflow as tf
import numpy as np
#from shapely import geometry
import os
import matplotlib.pyplot as plt
from config import *
import csv
#import statistics
#from cv import optical_flow

import cv2

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

def groom_groundfile(file: str):
	#normalize the ground file. Processing produces big numbers
	#the sim doesn't like big numbers
	scenario = "scenario" + file.split("_")[0][1:]
	matrix = []
	with open(os.path.join(DATA_DIR, scenario, file), 'r') as f:
		reader = csv.reader(f, delimiter=",")
		matrix = list(reader)
	norm_flag = False
	for r in range(len(matrix)):
		for c in range(len(matrix[r])):
			if float(matrix[r][c]) > 100:
				norm_flag = True
				break
	if norm_flag:
		# keep the distances small
		for r in range(len(matrix)):
			for c in range(len(matrix[r])):
				matrix[r][c] = float(matrix[r][c]) / 100

	#fix the lengths of the ground truths according to the frame_coords in the pkls
	# stem = file.replace("_ground.csv", "")
	# pkls = [f for f in os.listdir(os.path.join(DATA_DIR, scenario, )) if stem in f and f.endswith(".pkl")]
	#
	# new_matrix = []
	# for pkl in pkls:
	# 	frame_coords = pickle.load(open(os.path.join(DATA_DIR, scenario, pkl), "rb"))
	# 	num_frames = len(frame_coords)
	# 	#pkl = "s1_p1_cam1_coords.pkl" <- example
	# 	cam_number = int(pkl.split(".")[0].split("_")[2][3:])
	# 	orig_len = len(matrix[cam_number - 1])  # -1 because cams start at 1
	# 	print(cam_number)
	# 	print(orig_len)
	# 	# map the gp to a new list
	# 	ground_truth_new_row = []
	# 	for i in range(num_frames):
	# 		gt_i = round((i/num_frames) * orig_len)
	# 		ground_truth_new_row.append(float(matrix[cam_number - 1][gt_i]))
	# 	new_matrix.append(ground_truth_new_row)
	#
	# matrix = new_matrix
	# update the file
	with open(os.path.join(DATA_DIR, scenario, file), 'w', newline='') as f:
		writer = csv.writer(f, delimiter=",")
		writer.writerows(matrix)


# dumbly get the matrix from file
# The file can be either the ground truth or the simulated distances
# each row corresponds to a camera, each column corresponds to a frame
# params: name of groundfile: sX_pY_ground.csv
# returns: matrix of ground truths. rows are cameras, columns are frames
def get_matrix(file: str):
	scenario = "scenario" + file.split("_")[0][-1]
	matrix = []
	with open(os.path.join(DATA_DIR, scenario, file), 'r') as f:
		reader = csv.reader(f, delimiter=",")
		matrix = list(reader)

	#convert everything to float
	for r, row in enumerate(matrix):
		for c, col in enumerate(row):
			if "ground" in file:
				matrix[r][c] = float(matrix[r][c])
			elif "fire" in file:
				matrix[r][c] = int(matrix[r][c])

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
# 	return ground_truth_new_size

def get_number_of_cameras(groundfile: str):
	with open(groundfile) as f:
		return len(f.readlines())

#
# literally just returns infinit (for use in creating default dicts)
#
def inf():
	return math.inf

def rescale_frame(frame, percent=25):
	width = int(frame.shape[1] * percent/ 100)
	height = int(frame.shape[0] * percent/ 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def convert_to_jpgs(video_path: str):
	path, video = os.path.split(video_path)
	video = video.split(".")[0]
	#the folder where the jpegs will go
	if not os.path.exists(os.path.join(path, video)):
		os.mkdir(os.path.join(path, video))

	vidcap = cv2.VideoCapture(video_path)

	success, image = vidcap.read()
	count = 0
	while success:
		image = rescale_frame(image, 25)
		cv2.imwrite(os.path.join(path, video, "frame%d.jpg" % count), image)  # save frame as JPEG file
		success, image = vidcap.read()
		print('Read a new frame: ', success)
		count += 1
import random
def tf_print(tensor, filename, new=False):
	if not new and os.path.exists(filename):
		os.remove(filename)
	if new and os.path.exists(filename):
		suffix = str(random.randint(0, 100000000))
		filename = filename + suffix
	return tf.print(tensor, output_stream="file://" + filename, summarize=-1)

# frame is a 3d matrix: pixels wide x pixels high x 3 (RBG)
def flatten_frame(frame):

	flattened = []
	for x in range(len(frame)):
		for y in range(len(frame[x])):
			for rgb in frame[x][y]:
				flattened.append(rgb)

	return flattened

if __name__ == "__main__":
	#os.chdir("SEC")
	#get_matrix("s2_p1_ground.csv")
	#groom_groundfile("s2_p1_ground.csv")
	convert_to_jpgs(os.path.join("SEC", DATA_DIR, "scenario2", "s2_p1_cam4.mp4"))