import os
from cv2 import compare

import tensorflow as tf
from tensorflow.python.keras.backend import flatten
from tensorflow.python.keras.engine import input_layer
from utils import *
from clash_checker import check_clashes
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

BATCH_SIZE = 4

def create_cam_ideal():
	cam1_ideal_fire = []
	cam2_ideal_fire = []

	for i in range(602):
		if i % 2 == 0:
			cam1_ideal_fire.append(1)
			cam2_ideal_fire.append(0)
		if i % 2 == 1:
			cam1_ideal_fire.append(0)
			cam2_ideal_fire.append(1)

	return cam1_ideal_fire, cam2_ideal_fire


def get_lstm_model():
	kernel_size = (3, 3)
	conv_to_LSTM_dims = (1,248,400,32)
	LSTM_to_conv_dims = (248,400,8)

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Input(name='the_input', dtype='float32',batch_shape=(BATCH_SIZE, 248, 400, 3)))
	model.add(tf.keras.layers.Conv2D(8, kernel_size, padding='same',
					activation='tanh', kernel_initializer='he_normal',
						name='conv1'))
	model.add(tf.keras.layers.Conv2D(16, kernel_size, padding='same',
					activation='tanh', kernel_initializer='he_normal',
						name='conv1_1'))
	model.add(tf.keras.layers.Conv2D(32, kernel_size, padding='same',
					activation='tanh', kernel_initializer='he_normal',
						name='conv1_2'))
	model.add(tf.keras.layers.Flatten())
	#model.add(tf.keras.layers.Reshape(target_shape=conv_to_LSTM_dims, name='reshapeconvtolstm'))
	#model.add(tf.keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3),
	#				input_shape=(None, 248, 400, 32),
	#				padding='same', return_sequences=True,  stateful=False))
	#model.add(tf.keras.layers.Reshape(target_shape=LSTM_to_conv_dims, name='reshapelstmtoconv'))
	#model.add(tf.keras.layers.Conv2D(1, (1,1), padding='same',
	#				activation='sigmoid', kernel_initializer='he_normal',
	#					name='decoder'))
	model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="to_fire"))
	model.summary()

	return model

# ignore the stuff below. this is just ripped off from the MSNIST classifier
def get_uncompiled_model(sample_frame):
	#frame = flatten_frame(sample_frame)

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=sample_frame.shape))
	model.add(tf.keras.layers.MaxPool2D((2,2)))
	model.add(tf.keras.layers.Convolution2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPool2D((2,2)))
	model.add(tf.keras.layers.Convolution2D(8, kernel_size=(3, 3), activation='relu'))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.SimpleRNN(128, return_sequences=True)) # input_shape=(4, 602)))#input_shape=[246, 398, 8])) 
	model.add(tf.keras.layers.Dense(1, name="to_fire"))
	#model.summary()

	return model


def get_compiled_model():
	model = get_lstm_model()
	model.compile(
		optimizer="rmsprop",
		loss=tf.keras.losses.BinaryCrossentropy(),
	)
	return model

model = None
def classify(frames):
	global model
	if model == None:
		model = get_compiled_model(frames[0])
	return model.predict(frames)

def get_frame(pics_path, frame_number):
	frame_name = "frame" + str(frame_number) + ".jpg"
	frame = cv2.imread(os.path.join(pics_path, frame_name))
	return frame


# ground_file = "sX_pY_ground.csv"
def train(ground_file):
	global model
	assert model
	scenario = "scenario1"
	#ground_truth_matrix = get_matrix(ground_file)
#
	#cam1_ground_truth = ground_truth_matrix[0][:602]
	#cam2_ground_truth = ground_truth_matrix[1]
	#ground_truth_matrix = [cam1_ground_truth, cam2_ground_truth]

	cam1_ideal_fire, cam2_ideal_fire = create_cam_ideal()

	ideal_fire_matrix = [cam1_ideal_fire, cam2_ideal_fire]

	all_frames = []
	i = 0
	for row_i in range(len(ideal_fire_matrix)):
		row = ideal_fire_matrix[row_i]
		cam_name = "cam" + str(row_i + 1)
		cam_pics = os.path.join(DATA_DIR, scenario, "s1_p1_" + cam_name)
		frames = []
		for col_i in range(len(row)):
			frame = get_frame(cam_pics, col_i)
			frames.append(frame)
			i += 1
			if i == 602:
				break
		i = 0
		all_frames += frames

	ideal_fire_list = cam1_ideal_fire + cam2_ideal_fire
	tf_data = tf.data.Dataset.from_tensor_slices((all_frames, ideal_fire_list))
	tf_data = tf_data.batch(BATCH_SIZE)
	model.fit(tf_data, batch_size=BATCH_SIZE, verbose=1, epochs=20)

	testing_data = tf.data.Dataset.from_tensor_slices(all_frames)
	testing_data = testing_data.batch(BATCH_SIZE)
	compare_to_ideal(classify(testing_data))

def predict_fire_tf(jpgs_folder):
	frame_list = []
	once = True
	for frame in os.listdir(jpgs_folder):
		frame = cv2.imread(os.path.join(jpgs_folder, frame))
		if once:
			#print(frame)
			print(frame.shape)
			exit(0)
		once = False
		frame_list.append(frame)
	outcome = classify(frame_list)
	return outcome

def compare_to_ideal(classification):
	cam1_ideal, cam2_ideal = create_cam_ideal()
	ideals = cam1_ideal + cam2_ideal
	both = zip(ideals, classification)
	diffs = [abs(x - y) for x, y in both]

	print(" ideal | class | diff ")
	print("----------------------")
	for i in range(len(diffs)):
		print(ideals[i], "|", classification[i][0], "|", diffs[i][0], "\n")

	#ids = np.arange(len(diffs))
	#plt.plot(ids[50], diffs[50])
	#plt.show()

if __name__ == "__main__":
	os.chdir("SEC")
	sample_frame = cv2.imread(os.path.join("data", "scenario1", "s1_p1_cam1", "frame0.jpg"))
	model = get_compiled_model()
	train("s1_p1_ground.csv")




