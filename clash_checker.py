import os

from utils import get_matrix, tf_print
import numpy as np
import csv
from tensorflow.python.framework.ops import get_gradient_function

import tensorflow as tf

# replaces the matlab simulator
# can give either matrix directly or
def check_clashes(ground_matrix=None, fire_matrix=None, ground_path=None, fire_path=None, output_path=None):
    # if type(ground_matrix) == type(None):
    #     assert ground_path
    #     ground_matrix = np.array(get_matrix(ground_path))
    # if type(fire_matrix) == type(None):
    #     assert fire_path
    #     fire_matrix = np.array(get_matrix(fire_path))

    #tf_print(fire_matrix, "fire_matrix")

    # # one row where each element specifies
    # # the number of cameras that fire in that frame
    # collisions = tf.raw_ops.Greater(x=fire_matrix, y=0.5)
    #
    # # for each column, were there any collisions?
    # collisions = tf.reduce_any(collisions, axis=0)
    #
    # # but this is a 1D tensor. we want a 2D tensor so this works on the fire order
    # # so we just repeat this row by the n_cams
    # num_cams = tf.constant(4)
    # collisions = tf.repeat([collisions], repeats=[num_cams], axis=0)
    #
    # # 0 if collision or non fire
    # estimated_depths = tf.where((collisions == False) & tf.raw_ops.Greater(x=fire_matrix, y=0.5),
    #     ground_matrix, tf.zeros((num_cams, tf.constant(1))))

    estimated_depths = tf.math.multiply(fire_matrix, ground_matrix)
    tf_print(estimated_depths, "estimates")

    # if output_path:
    #     with open(output_path, "w", newline='') as f:
    #         csvwriter = csv.writer(f, delimiter=",")
    #         csvwriter.writerows(output_matrix)
    # no need to read the file too
    return estimated_depths
