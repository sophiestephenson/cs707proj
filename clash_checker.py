import os

from utils import get_matrix, tf_print
import numpy as np
import csv
import tensorflow as tf

# replaces the matlab simulator
# can give either matrix directly or
def check_clashes(ground_matrix=None, fire_matrix=None, ground_path=None, fire_path=None, output_path=None):
    if type(ground_matrix) == type(None):
        assert ground_path
        ground_matrix = np.array(get_matrix(ground_path))
    if type(fire_matrix) == type(None):
        assert fire_path
        fire_matrix = np.array(get_matrix(fire_path))

    #tf_print(fire_matrix, "fire_matrix")

    # one row where each element specifies
    # the number of cameras that fire in that frame
    #collisions = np.sum(fire_matrix, axis=0)
    collisions = tf.math.reduce_sum(tf.math.round(fire_matrix), axis=0)

    collision_threshold = tf.constant([1.0])
    # collisions is a list of booleans [True, False...]
    collisions = tf.math.greater(collisions, collision_threshold)
    # but this is a 1D tensor. we want a 2D tensor so this works on the fire order
    # so we just repeat this row by the n_cams
    collisions = tf.repeat([collisions], repeats=[fire_matrix.shape[0]], axis=0)
    #tf_print(collisions, "collisions_raw")
    # buuut we don't want to affect the rows that didn't fire
    # this is a list of indices to set to 0
    collisions = tf.where((collisions == True) & tf.greater(fire_matrix, 0.5))
    #tf_print(collisions, "collisions")
    # if a != None:
    #     with tf.compat.v1.Session() as sess:
    #         a.run(session=sess)
    # exit(0)
    #
    #
    #
    # output_matrix = ground_matrix.copy()
    # output_matrix = []

    # for r in range(len(ground_matrix)):
    #     row = ground_matrix[r]
    #     output_row = []
    #     last_d_est = 0.0
    #     for c in range(len(row)):
    #         depth = row[c]
    #         fire = fire_matrix[r, c]
    #         print(fire)
    #         d_est = 0.0
    #         if tf.math.greater():
    #
    #             # check if more than one camera fired in this frame
    #             if collisions[c] > 1.0:
    #                 # d_est = 0 by default
    #                 pass
    #             else:
    #                 d_est = depth
    #         # fire == 0
    #         else:
    #             d_est = last_d_est
    #         output_row.append(d_est)
    #     output_row = tf.convert_to_tensor(output_row)
    #     output_matrix.append(output_row)
    #
    # if output_path:
    #     with open(output_path, "w", newline='') as f:
    #         csvwriter = csv.writer(f, delimiter=",")
    #         csvwriter.writerows(output_matrix)
    # no need to read the file too
    #return tf.convert_to_tensor(output_matrix)
    return ground_matrix
