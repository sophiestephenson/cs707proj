################################
# utils.py
################################

import numpy as np

#
# given two coordinates (x, y), calculate the distance between them
#
# params: old coords, new coords
# returns: the distance between them: sqrt((x1 - x2)^2 + (y1 - y2)^2)
#
def calc_change(old_coords, new_coords):
	x1, y1 = old_coords
	x2, y2 = new_coords
	return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)