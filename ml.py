import os

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# y_true is an array of ground truth values
# y_pred is an array of predictions
def custom_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

# ignore the stuff below. this is just ripped off from the MSNIST classifier
def get_uncompiled_model(frame):
    inputs = keras.Input(shape=(frame.shape[0], frame.shape[1], 3), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model(frame):
    model = get_uncompiled_model(frame)
    model.compile(
        optimizer="rmsprop",
        loss=custom_loss_function
    )
    return model

model = None
def classify(frames):
    global model
    if model == None:
        model = get_compiled_model(frames[0])
    return model(frames)

def predict_fire_tf(jpgs_folder):
    frame_list = []
    once = True
    for frame in os.listdir(jpgs_folder):
        frame = cv2.imread(os.path.join(jpgs_folder, frame))
        if once:
            print(frame)
        once = False
        frame_list.append(frame)
    return classify(frame_list)

