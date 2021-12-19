import os
import sys
from utils import *

def make_new_scenario():
    path = os.path.join("SEC", DATA_DIR)
    scenarios = [x for x in os.listdir(path) if "scenario" in x]
    new_number = sorted([int(''.join(filter(str.isdigit, x))) for x in scenarios])[-1] + 1
    new_path = os.path.join(path, "scenario" + str(new_number))
    print("new path is:" + new_path)
    os.mkdir(new_path)
    return new_path, new_number


def explore(path):
    for dir in os.listdir(path):
        subpath = os.path.join(path, dir)
        print("path: " + dir)
        option = input("c for copy, e for explore, s for skip, q for quit: ")

        if option == "s":
            continue
        elif option == "e":
            explore(subpath)
        elif option == "c":
            new_path, number = make_new_scenario()
            stub = "s" + str(number) + "_p1_"
            with open(os.path.join(subpath, "row_per_cam.csv"), "r") as oldgroundfile:
                with open(os.path.join(new_path, stub + "ground.csv"), "w", newline='') as newgroundfile:
                    reader = csv.reader(oldgroundfile, delimiter=",")
                    matrix = np.array(list(reader))
                    # shave off headers
                    matrix = matrix[1:, 1:]
                    writer = csv.writer(newgroundfile, delimiter=",")
                    writer.writerows(matrix)

            dest = os.path.join(new_path, stub + "cam")
            for i, video_path in enumerate([os.path.join(subpath, v) for v in os.listdir(subpath) if "mp4" in v]):
                frames = convert_to_jpgs(video_path, dest + str(i+1))
                print("copied " + str(frames) + " frames")

        elif option == "q":
            return

if __name__ == "__main__":
    pass
    # path = "Data"
    # explore(path)

    # os.chdir("SEC")
    # for groundfile in get_list_of_gts():
    #     groom_groundfile(groundfile)
