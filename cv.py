################################
# cv.py
################################

from typing import Tuple
import cv2 as cv
import argparse
import numpy as np
from utils import *

#################################
# WRAPPERS FOR USE IN BLACK BOX
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
		print(np.mean(diffs))

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
		print(s)
		sizes.append(s)

	return sizes




def gather_data(filename):
	# get video
	capture = cv.VideoCapture(cv.samples.findFileOrKeep("cubes/two_cameras/" + filename))
	if not capture.isOpened():
		print("Unable to open", filename)
		exit(0)

	frame_coords = optical_flow(capture)
	#speeds = get_speeds(frame_coords)
	sizes = get_sizes(frame_coords)
	return sizes




#################################
# RAW CV FUNCTIONS
#################################


#
# uses optical flow CV to identify the change in position per frame and therefore
# the speed of movement between different frames.
#
# params: capture (the cv.VideoCapture object of the video)
# returns: an array of the speeds between each frame (for use in get_speeds)
#
def optical_flow(capture):
	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
						qualityLevel = 0.7,
						minDistance = 7,
						blockSize = 7 )
	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
					maxLevel = 2,
					criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
	# Create some random colors
	color = np.random.randint(0,255,(100,3))
	# Take first frame and find corners in it
	ret, old_frame = capture.read()
	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_frame)

	# keep track of the speeds as we go through the frames
	frames = []
	add_old = True

	while(1):
		ret,frame = capture.read()
		try:
			frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		except:
			print("color error, quitting")
			return frames
		# calculate optical flow
		p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

		# Select good points
		if p1 is not None:
			good_new = p1[st!=2]  # [st==1]
			good_old = p0[st!=2]  # [st==1]
			if add_old:
				frames.append(good_old)
				print(good_old)
				add_old = False
			frames.append(good_new)
			print(good_new)


		# draw the tracks
		for i,(new,old) in enumerate(zip(good_new, good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
			frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
		img = cv.add(frame,mask)
		cv.imshow('frame',img)
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
		# Now update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)

	#return speeds
	return frames


#
# shows the background-subtracted version of the video
#
# params: capture (the cv.VideoCapture object of the video)
#
def background_subtractor(capture):
	backSub = cv.createBackgroundSubtractorKNN()

	while True:
		ret, frame = capture.read()
		if frame is None:
			break

		fgMask = backSub.apply(frame)

		cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
		cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
					cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
		
		
		cv.imshow('Frame', frame)
		cv.imshow('FG Mask', fgMask)
		
		keyboard = cv.waitKey(30)
		if keyboard == 'q' or keyboard == 27:
			break


#
# shows the meanshift-tracker version of the video
# DOES NOT CURRENTLY WORK
#
# params: capture (the cv.VideoCapture object of the video)
#
def meanshift_tracker(capture):
	# take first frame of the video
	ret,frame = capture.read()
	# setup initial location of window
	x, y, w, h = 1300, 350, 300, 300 # simply hardcoded the values
	track_window = (x, y, w, h)

	# set up the ROI for tracking
	roi = frame[y:y+h, x:x+w]
	hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
	mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
	roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

	# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
	term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
	while(1):
		ret, frame = capture.read()
		if ret == True:
			hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
			dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
			# apply meanshift to get the new location
			ret, track_window = cv.meanShift(dst, track_window, term_crit)
			# Draw it on image
			x,y,w,h = track_window
			img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
			cv.imshow('img2',img2)
			k = cv.waitKey(30) & 0xff
			if k == 27:
				break
		else:
			break

#################################
# MAIN
#################################

#
# default function if the file is run
# takes command line args -f [filename] [-bs] [-mt] [-of]
#
def main():
	# parse args
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', type=str, help='Filepath to the video', default='cubes/two_cameras/cam1.mov')
	parser.add_argument('-bs', action='store_true', help='Run background subtractor (opt)')
	parser.add_argument('-mt', action='store_true', help='Run meanshift tracker (opt)')
	parser.add_argument('-of', action='store_true', help='Run optical flow (opt)')
	args = parser.parse_args()

	# get video
	capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.f))
	if not capture.isOpened():
		print("Unable to open", args.f)
		exit(0)

	# run program
	if args.mt:
		meanshift_tracker(capture)
	if args.bs: 
		background_subtractor(capture)
	if args.of:
		optical_flow(capture)

main()




