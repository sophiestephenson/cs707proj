import os
#from cv2 import _InputArray_KIND_SHIFT

import tensorflow as tf
from tensorflow.python.keras.backend import flatten
from tensorflow.python.keras.engine import input_layer
from utils import *
from clash_checker import check_clashes
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

SLICE_SIZE = 4
import sys

ground_truths = {}

current_n_cam = 4
# y_true is an array of ground truth values
# y_pred is an array of predictions
def custom_loss_function(ground_truth_matrix: tf.Tensor, predicted_fire_matrix: tf.Tensor):
    global current_n_cam
    #print("IN LOSS")
    #tf.print(predicted_fire_matrix, output_stream=sys.stdout)
    #tf.print(ground_truth_matrix)
    # reshaped_fire = predicted_fire_matrix.reshape((current_n_cam, len(predicted_fire_matrix)//current_n_cam))
    # reshaped_ground = ground_truth_matrix.reshape((current_n_cam, len(predicted_fire_matrix)//current_n_cam))
    # depth_estimates = check_clashes(ground_matrix=reshaped_ground, fire_matrix=reshaped_fire)
    # depth_estimates = tf.reshape(depth_estimates, ground_truth_matrix.shape)
    tf_print(ground_truth_matrix, "gt")
    tf_print(predicted_fire_matrix, "pred")
    retval = tf.square(tf.math.subtract(ground_truth_matrix, predicted_fire_matrix))
    tf_print(retval, "mean")
    return retval
    #return tf.repeat([500.0], repeats=[predicted_fire_matrix.shape[0]])

# ignore the stuff below. this is just ripped off from the MSNIST classifier
def get_uncompiled_model(sample_frame):
    #frame = flatten_frame(sample_frame)
    model = tf.keras.Sequential()
    print(sample_frame.shape)
    # conv_layer = tf.keras.layers.Convolution2D(32, kernel_size=(5, 5),
    #     strides=(1,1), input_shape=sample_frame.shape)
    # model.add(tf.keras.layers.TimeDistributed(conv_layer,
    #     input_shape=(SLICE_SIZE, sample_frame.shape[0], sample_frame.shape[1], sample_frame.shape[2])))
    # pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
    # model.add(tf.keras.layers.TimeDistributed(pooling_layer))
    # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, input_shape=(1,)),
    #                                           input_shape=(SLICE_SIZE,1)))
    #model.add(tf.keras.layers.SimpleRNN(128)) #, input_shape=(None, SLICE_SIZE, 32000)))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(1, name="to_fire"))
    model.build(input_shape=(1,1))
    print(model.summary())


    #inputs = keras.Input(shape=(frame.shape[0], frame.shape[1], 3), name="digits")
    #x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    #x = layers.Dense(64, activation="relu", name="dense_2")(x)
    #outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    #model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model(sample_frame):
    model = get_uncompiled_model(sample_frame)
    model.compile(
        optimizer="adam",
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

def get_slices(stem, limit=1, cams=None):
    scenario = "scenario" + stem.split("_")[0][-1]
    assert scenario == "scenario2"
    path = os.path.join(DATA_DIR, scenario)
    cam_folders = [os.path.join(path, d) for d in os.listdir(path) if stem in d and os.path.isdir(os.path.join(path, d))]
    n_frames = len(os.listdir(cam_folders[0]))
    n_cams = len(cam_folders)


    all_frames = []
    ele_shape = None
    r = 0
    for row_i in range(n_cams):
        cam_name = "cam" + str(row_i + 1)
        if cams and cam_name not in cams:
            continue
        cam_pics = os.path.join(path, stem + cam_name)
        frames_slice = []
        all_frames.append([])
        for col_i in range(n_frames - 1):
            frame = get_frame(cam_pics, col_i)
            if type(frame) == type(None):
                continue
            frames_slice.append(frame)
            if len(frames_slice) == SLICE_SIZE:
                frame_slice_nd = np.array(frames_slice)
                if not ele_shape:
                    ele_shape = frame_slice_nd.shape
                else:
                    if not ele_shape == frame_slice_nd.shape:
                        assert ele_shape == frame_slice_nd.shape
                all_frames[r].append(frame_slice_nd)
                # remove oldest frame
                frames_slice.pop(0)
        r += 1
        #all_frames.append(frames)

    #pass frames in column order. (cam1, frame0), (cam2, frame0)...
    flat_frame_list = []
    for col_i in range(len(all_frames[0])):
        for cam_i in range(len(all_frames)):
            flat_frame_list.append(all_frames[cam_i][col_i])
    flat_frame_list = np.array(flat_frame_list)
    if cams:
        batch = len(cams)
    else:
        batch = n_cams
    return flat_frame_list[:int(batch * limit)]

# ground_file = "sX_pY_ground.csv"
def train(ground_file):
    global model
    assert model
    scenario = "scenario" + ground_file.split("_")[0][-1]
    assert scenario == "scenario2"
    #ground_path = os.path.join(DATA_DIR, scenario, ground_file)
    ground_truth_matrix = np.array(get_matrix(ground_file), dtype=float)
    stem = ground_file.replace('ground.csv', '')

    flat_frame_list = get_slices(stem) # limit=1

    # 0, 1, 2 cannot have 4 frame slices. also, for some reason GT has one extra frame
    ground_truth_matrix = ground_truth_matrix[:, SLICE_SIZE:]

    #pass ground truths in column order. (cam1, frame0), (cam2, frame0)...
    gt_flat = []
    for col_i in range(len(ground_truth_matrix[0])):
        for cam_i in range(len(ground_truth_matrix)):
            gt_flat.append(ground_truth_matrix[cam_i][col_i])
    gt_flat = np.array(gt_flat)
    #TODO: Can we fit a 3D matrix?
    #tf_print(tf.convert_to_tensor(flat_frame_list[:4]), "training_data")
    #model.fit(flat_frame_list[:4], gt_flat[:4], batch_size=4, epochs=10, shuffle=False)

    fake_data = np.reshape(gt_flat, (gt_flat.shape[0],1))
    model.fit(fake_data, gt_flat, epochs=12, batch_size=4, shuffle=False)


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

def predict_fire_tf(stem, cam, limit=1):
    #frame_slice_list = get_slices(stem, limit, [cam])
    dummy_list = get_matrix(stem + "ground.csv")

    row = int(cam[-1]) - 1
    dummy_list = np.reshape(np.array(dummy_list[row]), (len(dummy_list[row]),1))
    # frame_slice = []
    # for frame in os.listdir(jpgs_folder):
    #     frame = cv2.imread(os.path.join(jpgs_folder, frame))
    #     frame_slice.append(frame)
    #     if len(frame_slice) == SLICE_SIZE:
    #         frame_slice_list.append(tf.convert_to_tensor(frame_slice))
    #         frame_slice.pop(0)
    #         break
    #predict_tensor = tf.convert_to_tensor(frame_slice_list)
    predict_tensor = tf.convert_to_tensor(dummy_list)
    tf_print(predict_tensor, "predict_input_data", new=True)
    outcome = classify(predict_tensor)
    return outcome

new_model = False
if __name__ == "__main__":
    os.chdir("SEC")
    if not new_model:
        try:
            model = tf.keras.models.load_model("model_save", compile=False)
        except IOError:
            model = None
    else:
        model = None
    sample_frame = cv2.imread(os.path.join("data", "scenario2", "s2_p1_cam1", "frame0.jpg"))
    if model == None:
        print("generating model")
        model = get_compiled_model(sample_frame)
        train("s2_p1_ground.csv")
        model.save("model_save")


    # retval = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam1"))
    # retval2 = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam2"))
    # retval3 = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam3"))
    # retval4 = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam4"))

    retval = predict_fire_tf("s2_p1_", "cam1")
    retval2 = predict_fire_tf("s2_p1_", "cam2")
    retval3 = predict_fire_tf("s2_p1_", "cam3")
    retval4 = predict_fire_tf("s2_p1_", "cam4")
    # summarize = -1 means print the full tensor
    tf_print(retval, "outfile1")
    tf_print(retval2, "outfile2")
    tf_print(retval3, "outfile3")
    tf_print(retval4, "outfile4")


