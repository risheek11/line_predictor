"""
    Project: Initial setup for Data Processing methods.
    Author: Goel, Ayush
    Date: 1st August 2019
"""


from os import listdir
from utilities import read_json_file, create_dir, LOGGER, MIN_DIM
from utilities.config import PROCESSED_DATA_DIR, BBOX_IMAGES_PATH, RAW_TRAIN_IMAGES_PATH, IMG_INFO_JSON_PATH, \
    NORMALISED_IMAGES_PATH, NORMALISED_BBOX_IMAGES_PATH


IMG_INFO_JSON = read_json_file(IMG_INFO_JSON_PATH)

ANNOTATIONS = IMG_INFO_JSON['annotations']
IMAGES = IMG_INFO_JSON['images']

IMAGES_IN_DIR = listdir(RAW_TRAIN_IMAGES_PATH)
create_dir(PROCESSED_DATA_DIR)
