import os

import numpy as np

class OutputVector:

    # time is expressed in number of frames, ie integers
    def __init__(self, p_mod: float, sleep_time: int, fire_time: int):
        self.p_mod = p_mod
        self.sleep_time = sleep_time
        self.fire_time = fire_time



# I don't think this function is actually going to get used
# but it can be used as a model for the sim equivalent
def emit_list(vector_list: list[OutputVector]):
    retval = []
    fire_countdown = []
    sleep_countdown = -1
    for v in vector_list:
        if v.sleep_time:
            # don't override any previous sleep mandate
            sleep_countdown = max(sleep_countdown, v.sleep_time)
        if v.fire_time:
            # we will save all fire opportunities
            fire_countdown.append(v.fire_time)

        # do we fire?
        # yes, if no sleep and we have a fire opportunity
        if sleep_countdown <= 0 and 0 in fire_countdown:
            retval.append(1)
        # definite no if we are sleeping
        elif sleep_countdown > 0:
            retval.append(0)
        else:
            retval.append(v.p_mod)

        # 1 frame has passed
        sleep_countdown -= 1
        for i in range(len(fire_countdown)):
            fire_countdown[i] -= 1

    return retval

"""
Takes in *nothing* but an RGB frame filepath
Returns an OutputVector
"""
def predict(frame_path):
    return OutputVector(0.5, 0, 0)


if __name__ == '__main__':
    FRAMES_FOLDER = "rgb_images"
    OUTFILE = "vector.matrix"
    cameras = sorted(os.listdir(FRAMES_FOLDER))
    # 0 stands for camera 0
    frames = os.listdir(os.path.join(FRAMES_FOLDER, "0"))

    # rows are cameras and columns are frames
    # so for the basic case, we expect 2 rows and f columns (f being the total number of frames)
    output_matrix = np.ndarray(shape=(len(cameras), len(frames)), dtype=OutputVector)

    for f_counter, frame_name in enumerate(frames):
        for camera in range(0, len(cameras)):
            # add this prediction to the matrix
            ov = predict(os.path.join(FRAMES_FOLDER), frame_name)
            output_matrix[camera][f_counter] = ov

    with open(OUTFILE, "w") as f:
        output_matrix.tofile(fid=f, sep=",")

