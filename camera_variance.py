from utils import *

if __name__ == "__main__":
    camera_depths= {"1": [], "2": [], "3": [], "4": []}
    ground_files = get_list_of_gts()
    for gf in ground_files:
        matrix = get_matrix(gf)
