################################
# blackbox.py
################################

from cv import get_speeds

# pipeline
# 	1. for each camera:
#   	- read info from video (speed, direction, etc.)
#   	- predict when to fire
#   2. send prediction information to the simulator
#   3. use feedback from the simulator to update predictions

def read_rbg_frame():

	# how fast is the scene changing?
	speeds = get_speeds("cam2.mov")

	# how big is the object?
	# what direction is the object moving in?
	# TODO: can probably estimate these given the positions of interest in optical flow

	return

def predict_fire():
	return 1



read_rbg_frame()