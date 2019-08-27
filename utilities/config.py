"""
    Configuration file reader for DHCP
    Reads configs.ini and loads into corresponding variables

    Author: Goel, Ayush
    Date Created: 15th July 2019
"""

import configparser
import json

CONFIGS = configparser.ConfigParser()
CONFIGS.read('utilities/config.ini')

WORK_DIR = ''

# LOGGER
LOG_LEVEL = CONFIGS['LOGGER']['LOG_LEVEL']
LOGS_DIR = CONFIGS['LOGGER']['LOGS_DIR']


# PRE PROCESSING
PROCESSED_DATA_DIR = CONFIGS['PROCESSING']['PROCESSED_DATA_DIR']

IMG_INFO_JSON_PATH = WORK_DIR + CONFIGS['DATASET']['IMG_INFO_JSON_PATH']
RAW_TRAIN_IMAGES_PATH = CONFIGS['DATASET']['RAW_TRAIN_IMAGES_PATH']


BBOX_IMAGES_PATH = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['BBOX_IMAGES_PATH']
MIN_DIM = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['MIN_DIM']

NORMALISED_IMAGES_PATH = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['NORMALISED_IMAGES_PATH']
NORMALISED_BBOX_IMAGES_PATH = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['NORMALISED_BBOX_IMAGES_PATH']
BBOX_CSV_PATH = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['BBOX_CSV_PATH']
BBOX_JSON_PATH = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['BBOX_JSON_PATH']
NORM_BBOX_JSON_PATH = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['NORM_BBOX_JSON_PATH']
BBOX_XYWH_JSON_PATH = WORK_DIR + PROCESSED_DATA_DIR + CONFIGS['PROCESSING']['BBOX_XYWH_JSON_PATH']

# RPN
ASPECT_RATIOS = json.loads(CONFIGS['RPN']['ASPECT_RATIOS'])
SCALES = json.loads(CONFIGS['RPN']['SCALES'])
N_ANCHOR_BOXES = int(len(ASPECT_RATIOS) * len(SCALES))
REG_HIDDEN_LAYERS = int(CONFIGS['RPN']['REG_HIDDEN_LAYERS'])
CLS_HIDDEN_LAYERS = int(CONFIGS['RPN']['CLS_HIDDEN_LAYERS'])

# VGG
VGG_CHANNELS = int(CONFIGS['VGG']['VGG_CHANNELS'])
VGG_SCALE_SIZE = int(CONFIGS['VGG']['VGG_SCALE_SIZE'])

# TRAINING
EPOCHS = int(CONFIGS['TRAINING']['EPOCHS'])
LEARNING_RATE = float(CONFIGS['TRAINING']['LEARNING_RATE'])
MOMENTUM = float(CONFIGS['TRAINING']['MOMENTUM'])
SCHEDULER_GAMMA = float(CONFIGS['TRAINING']['SCHEDULER_GAMMA'])
SCHEDULER_STEP = int(CONFIGS['TRAINING']['SCHEDULER_STEP'])
EPOCH_SAVE_INTERVAL = int(CONFIGS['TRAINING']['EPOCH_SAVE_INTERVAL'])
MODEL_SAVE_PATH = CONFIGS['TRAINING']['MODEL_SAVE_PATH']


# TESTING
TEST_IMAGES_PATH = CONFIGS['TESTING']['TEST_IMAGES_PATH']
TEST_OUTPUT_PATH = CONFIGS['TESTING']['TEST_OUTPUT_PATH']
TOP_ANCHORS = int(CONFIGS['TESTING']['TOP_ANCHORS'])
OVERLAP_THRESHOLD = float(CONFIGS['TESTING']['OVERLAP_THRESHOLD'])