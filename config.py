################################
# config.py
################################

# general configs
# ../ needed because the working dir is ./SEC/
DIRECTORY = "../cubes/two_cameras/"
DATA_DIR = "data"

# reinforcement learning
SIZE_FACTOR_RANGE = 20000
SPEED_FACTOR_RANGE = 20
DIRECTION_FACTOR_RANGE = 20
WAIT_TIME_MAX = 20

# cv configs
OPTICAL_FLOW_QUALITY = 0.7
MIN_DISTANCE = 7
BLOCK_SIZE = 10

# smoothing
SMOOTHING_KERNEL_SIZE = 20