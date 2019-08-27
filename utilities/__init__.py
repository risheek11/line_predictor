"""
    Project: Utility Functions Init File
    Author: Goel, Ayush
    Date: 1st August 2019
"""
import json
import logging
import os
from datetime import datetime
from .config import LOG_LEVEL, MIN_DIM, RAW_TRAIN_IMAGES_PATH, PROCESSED_DATA_DIR


def create_dir(dirname):
    """
    Creates a directory if it does not exist
    :param dirname:
    :return: None
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def check_path(path):
    """
    Check if it a valid path.
    :param path:
    :return: Boolean for existence of path.
    """
    return os.path.isfile(path)


def create_logger(logger_name, file_name):
    """
    :param logger_name:
    :param file_name:
    :return: logger with logger_name as specified and writes to file
    """

    create_dir('logs/')
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)-4d] [%(funcName)s] %(message)s"
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL)

    handler = logging.FileHandler('logs/' + file_name + '.log')
    console = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    console.setLevel('INFO')
    handler.setFormatter(formatter)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


LOGGER = create_logger('LOGGER', datetime.now().strftime("%Y-%m-%d"))


def read_json_file(filename):
    """
    :param filename:
    :return:
    """
    with open(filename, encoding='utf-8') as json_file:
        json_object = json.load(json_file)
    return json_object


def write_json_to_file(jsonobj, filename):
    """
    :param jsonobj:
    :param filename:
    :return:
    """
    with open(filename, 'w', encoding='utf8') as file:
        json.dump(jsonobj, file, indent=4)
    return jsonobj, filename


def get_min_dims_to_resize():
    import cv2
    import pickle
    create_dir(PROCESSED_DATA_DIR)
    IMAGES_IN_DIR = os.listdir(RAW_TRAIN_IMAGES_PATH)
    x = 9999999999
    y = 9999999999
    for image in IMAGES_IN_DIR:
        img = cv2.imread(RAW_TRAIN_IMAGES_PATH + image)
        y_ = img.shape[0]
        x_ = img.shape[1]
        if x > x_:
            x = x_
        if y > y_:
            y = y_
    minimum = min(x, y)
    LOGGER.debug('Minimum dim found as %d', 16 * (minimum // 16))
    fp = open(MIN_DIM, 'wb')
    pickle.dump(16 * (minimum // 16), fp)
    fp.close()

get_min_dims_to_resize()
