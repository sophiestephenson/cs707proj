################################
# utils.py
################################

import numpy as np
from shapely import geometry

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


