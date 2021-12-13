import os

import tensorflow as tf
from utils import *
from clash_checker import check_clashes
import numpy as np

# y_true is an array of ground truth values
# y_pred is an array of predictions
def custom_loss_function(predicted_fire_matrix: np.ndarray, ground_truth_matrix: np.ndarray):
    depth_estimates = np.array(check_clashes(predicted_fire_matrix))
    return tf.math.reduce_mean(tf.square(ground_truth_matrix - depth_estimates))

# ignore the stuff below. this is just ripped off from the MSNIST classifier
def get_uncompiled_model(sample_frame):

    model = tf.keras.Sequential()
    model.add(tf.keras.SimpleRNN(128, input_shape=sample_frame.shape))
    model.add(tf.keras.Dense(1))

    #inputs = keras.Input(shape=(frame.shape[0], frame.shape[1], 3), name="digits")
    #x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    #x = layers.Dense(64, activation="relu", name="dense_2")(x)
    #outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    #model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model(sample_frame):
    model = get_uncompiled_model(sample_frame)
    model.compile(
        optimizer="rmsprop",
        loss=custom_loss_function
    )
    print("returning model")
    return model

model = None

def classify(frames):
    global model
    if model == None:
        model = get_compiled_model(frames[0])
    return model(frames)

def get_frame(pics_path, frame_number):
    frame_name = "frame" + str(frame_number) + ".jpg"
    frame = cv2.imread(os.path.join(pics_path, frame_name))
    return frame


# ground_file = "sX_pY_ground.csv"
def train(ground_file):
    global model
    assert model
    scenario = "scenario1"
    ground_path = os.path.join(DATA_DIR, scenario, ground_file)
    ground_truth_matrix = get_matrix(ground_path)

    all_frames = []
    for row_i in range(len(ground_truth_matrix)):
        row = ground_truth_matrix[row_i]
        cam_name = "cam" + str(row_i + 1)
        cam_pics = os.path.join(DATA_DIR, scenario, "s1_p1_" + cam_name)
        frames = []
        for col_i in range(len(row)):
            frame = get_frame(cam_pics, col_i)
            frames.append(frame)
        all_frames.append(frames)
    print("about to fit")
    #TODO: Can we fit a 3D matrix?
    model.fit(all_frames, ground_truth_matrix)

    # for row_i in range(len(ground_truth_matrix)):
    #     row = ground_truth_matrix[row_i]
    #     cam_name = "cam" + str(row_i + 1)
    #     cam_pics = os.path.join(DATA_DIR, scenario, "s1_p1_" + cam_name)
    #     frames = []
    #     should_fires = []
    #     for col_i in range(len(row)):
    #         should_fire = (col_i + row_i) % len(ground_truth_matrix)
    #         frame = get_frame(cam_pics, col_i)
    #         frames.append(frame)
    #         should_fires.append(should_fire)

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

if __name__ == "__main__":
    sample_frame = cv2.imread("SEC\data\scenario1\s1_p1_cam1\frame0.jpg")
    model = get_compiled_model(sample_frame)
    train("s1_p1_ground.csv")
