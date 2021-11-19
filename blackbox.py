################################
# blackbox.py
################################

from cv import gather_data
from pprint import pprint
import matplotlib.pyplot as plt
from utils import smooth_data, reject_outliers, plot

# pipeline
# 	1. for each camera:
#   	- read info from video (speed, direction, etc.)
#   	- predict when to fire
#   2. send prediction information to the simulator
#   3. use feedback from the simulator to update predictions

def read_rbg_frame():
	speeds, sizes, direction = gather_data("cam2")

	#plot(speeds, "rates of change")
	plot(smooth_data(speeds), "rates of change (smoothed)")
	#plot(sizes, "sizes")
	plot(smooth_data(sizes), "sizes (smoothed)")
	pprint(direction)

def predict_fire():
	return 1

read_rbg_frame()