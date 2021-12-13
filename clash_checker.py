from utils import get_matrix
import numpy as np
import csv


# replaces the matlab simulator
# can give either matrix directly or
def check_clashes(ground_matrix=None, fire_matrix=None, ground_path=None, fire_path=None, output_path=None):
    if not ground_matrix:
        assert ground_path
        ground_matrix = np.array(get_matrix(ground_path))
    if not fire_matrix:
        assert fire_path
        fire_matrix = np.array(get_matrix(fire_path))

    # one row where each element specifies
    # the number of cameras that fire in that frame
    collisions = np.sum(fire_matrix, axis=0)

    output_matrix = []
    for r, row in enumerate(ground_matrix):
        output_row = []
        last_d_est = 0
        for c, depth in enumerate(row):
            fire = fire_matrix[r, c]
            d_est = 0
            if fire:
                # check if more than one camera fired in this frame
                if collisions[c] > 1:
                    # d_est = 0 by default
                    pass
                else:
                    d_est = depth
            # fire == 1
            else:
                d_est = last_d_est
            output_row.append(d_est)
        output_matrix.append(output_row)

    if output_path:
        with open(output_path, "w", newline='') as f:
            csvwriter = csv.writer(f, delimiter=",")
            csvwriter.writerows(output_matrix)
    # no need to read the file too
    return output_matrix
