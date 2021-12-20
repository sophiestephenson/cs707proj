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

N_CAMS = 4

from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D
"""
from: https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
"""
def build_convnet(shape):
    momentum = .9
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=shape,
                     padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # flatten...
    model.add(GlobalMaxPool2D())
    return model

# y_true is an array of ground truth values
# y_pred is an array of predictions
def custom_loss_function(ground_truth_matrix: tf.Tensor, predicted_fire_matrix: tf.Tensor):
    global N_CAMS
    tensor_N_CAMS = tf.constant(4)
    tensor_FRAMES = tf.constant(1)
    print("IN LOSS")
    # new_length = tf.cast(tf.math.round(tf.math.divide(predicted_fire_matrix.shape[0], tensor_N_CAMS)), dtype=tf.int32)
    # tf_print(predicted_fire_matrix.shape[0], "oldlength")
    # print(new_length, "length")
    # new_shape = tf.convert_to_tensor([tensor_N_CAMS, new_length])
    # flatten the shape value to (rows, cols)
    # new_shape = tf.reshape(new_shape, [-1])
    new_shape = (tensor_N_CAMS, tensor_FRAMES)
    reshaped_fire = tf.reshape(predicted_fire_matrix, new_shape)
    reshaped_ground = tf.reshape(ground_truth_matrix, new_shape)
    depth_estimates = check_clashes(ground_matrix=reshaped_ground, fire_matrix=reshaped_fire)
    depth_estimates = tf.reshape(depth_estimates, ground_truth_matrix.shape)
    tf_print(ground_truth_matrix, "gt")
    tf_print(predicted_fire_matrix, "pred")
    retval = tf.square(tf.math.divide(tf.math.subtract(ground_truth_matrix, depth_estimates), ground_truth_matrix))\
        + tf.square(tf.subtract(1.0, tf.reduce_sum(predicted_fire_matrix)))
    tf_print(retval, "loss")
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
    convnet = build_convnet(sample_frame.shape)
    model.add(tf.keras.layers.TimeDistributed(convnet,
        input_shape=(SLICE_SIZE, sample_frame.shape[0],
        sample_frame.shape[1], sample_frame.shape[2])))
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128)
    ))
    model.add(tf.keras.layers.SimpleRNN(128))
    model.add(tf.keras.layers.Dense(1, name="distance"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.build(input_shape=(SLICE_SIZE, sample_frame.shape[0],
                             sample_frame.shape[1], sample_frame.shape[2]))
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
        loss="mean_squared_error",
        metrics="mean_absolute_percentage_error"
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

def get_slices(stem, limit=None, cams=None):
    scenario = "scenario" + stem.split("_")[0][1:]
    path = os.path.join(DATA_DIR, scenario)
    cam_folders = [os.path.join(path, d) for d in os.listdir(path) if stem in d and os.path.isdir(os.path.join(path, d))]
    n_frames = len(os.listdir(cam_folders[0]))
    n_cams = len(cam_folders)

    # only need this for fake
    ground_truth = np.array(get_matrix(stem + "ground.csv"))

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
        for col_i in range(n_frames-1):
            frame = get_frame(cam_pics, col_i)
            # THIS IS FAKE
            #frame = ground_truth[row_i, col_i]
            if type(frame) == type(None):
                continue
            frames_slice.append(frame)
            if len(frames_slice) == SLICE_SIZE:
                frame_slice_nd = np.array(frames_slice)
                if not ele_shape:
                    ele_shape = frame_slice_nd.shape
                else:
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
    if not limit:
        return flat_frame_list
    else:
        return flat_frame_list[:int(batch * limit)]

# ground_file = "sX_pY_ground.csv"
def train(ground_files):
    global model
    assert model

    training_data = []
    training_data_fake = []
    ground_truths = []
    for ground_file in ground_files:
        print(ground_file)
        scenario = "scenario" + ground_file.split("_")[0][1:]
        #ground_path = os.path.join(DATA_DIR, scenario, ground_file)
        ground_truth_matrix = np.array(get_matrix(ground_file), dtype=float)
        stem = ground_file.replace('ground.csv', '')
        flat_slice_list = get_slices(stem)

        # 0, 1, 2 cannot have 4 frame slices
        ground_truth_matrix = ground_truth_matrix[:, SLICE_SIZE-1:len(flat_slice_list)//N_CAMS + (SLICE_SIZE-1)]
        #pass ground truths in column order. (cam1, frame0), (cam2, frame0)...
        gt_flat = []
        for col_i in range(len(ground_truth_matrix[0])):
            for cam_i in range(len(ground_truth_matrix)):
                gt_flat.append(ground_truth_matrix[cam_i][col_i])
        gt_flat = np.array(gt_flat)
        ground_truths.append(gt_flat)
        training_data_fake.append(flat_slice_list)
    training_data_fake = np.concatenate(training_data_fake)
    print("training data shape is: ")
    print(training_data_fake.shape)
    ground_truths = np.concatenate(ground_truths)
    print("all ground truths shape is: " + str(ground_truths.shape))
    model.fit(training_data_fake, ground_truths, epochs=30,  shuffle=True)


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

def predict_fire_tf(stem, cam, limit=None):
    frame_slice_list = get_slices(stem, limit, [cam])
    #dummy_list = get_matrix(stem + "ground.csv")
    #row = int(cam[-1]) - 1
    predict_tensor = tf.convert_to_tensor(frame_slice_list)
    #tf_print(predict_tensor, "predict_input_data", new=True)
    outcome = classify(predict_tensor)
    return outcome

force_new_model = True
if __name__ == "__main__":
    os.chdir("SEC")
    if not force_new_model:
        try:
            model = tf.keras.models.load_model("model_save", compile=False)
        except IOError:
            model = None
    else:
        model = None
    sample_frame = cv2.imread(os.path.join("data", "scenario1", "s1_p1_cam1", "frame0.jpg"))
    if model == None:
        print("generating model")
        model = get_compiled_model(sample_frame)
        ground_files = get_list_of_gts()
        train(ground_files[:-2])
        model.save("model_save")


    # retval = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam1"))
    # retval2 = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam2"))
    # retval3 = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam3"))
    # retval4 = predict_fire_tf(os.path.join("data", "scenario2", "s2_p1_cam4"))

    for ground_file in ground_files[-2:]:
        stem = ground_file.replace("ground.csv", "")
        cams = ["cam" + str(num) for num in range(1, 5)]
        for cam in cams:
            retval = predict_fire_tf(stem, cam, limit=1)
            tf_print(retval, stem+cam+"_outfile")



