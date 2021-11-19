################################
# utils.py
################################

import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt
from config import SMOOTHING_KERNEL_SIZE

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
