import cv2
import numpy as np
import pickle
from data_processing import IMAGES, IMAGES_IN_DIR, ANNOTATIONS
from utilities import create_dir, LOGGER, write_json_to_file, read_json_file
from utilities.config import BBOX_IMAGES_PATH, BBOX_JSON_PATH, RAW_TRAIN_IMAGES_PATH,\
    BBOX_CSV_PATH, NORMALISED_IMAGES_PATH, NORMALISED_BBOX_IMAGES_PATH, \
    BBOX_XYWH_JSON_PATH, NORM_BBOX_JSON_PATH, MIN_DIM


def get_coors_from_annotation_by_id(img_id=0):
    for annotation in ANNOTATIONS:
        if annotation['image_id'] == img_id:
            tc_coor = (annotation['segmentation'][0][0],
                       annotation['segmentation'][0][1])
            bl_coor = (annotation['segmentation'][0][4],
                       annotation['segmentation'][0][5])
            return tc_coor, bl_coor


def get_img_bbox_coors(limit=9999999999999999999):
    create_dir(BBOX_IMAGES_PATH)
    LOGGER.debug('Getting image map with coordinates for Bounding Box')
    json_processed = dict()
    count = 0
    for image in IMAGES:
        if image['file_name'] in IMAGES_IN_DIR:
            image_id = image['id']
            tc_coor, bl_coor = get_coors_from_annotation_by_id(image_id)
            json_processed[image['file_name']] = [tc_coor, bl_coor]
            print(json_processed)
            count += 1
        if count >= limit:
            break
    write_json_to_file(json_processed, BBOX_JSON_PATH)


def get_norm_xywh_format():
    json_bbox = read_json_file(NORM_BBOX_JSON_PATH)
    xywh_bbox = dict()
    for key, value in json_bbox.items():
        xywh_bbox[key] = [[(value[2] + value[0]) // 2,
                          (value[3] + value[1]) // 2,
                          value[2] - value[0],
                          value[3] - value[1]]]
    write_json_to_file(xywh_bbox, BBOX_XYWH_JSON_PATH)


def create_bbox():
    LOGGER.debug('Creating Bounded Boxes')
    json_bbox = read_json_file(BBOX_JSON_PATH)
    for key, value in json_bbox.items():
        im_rd = cv2.imread(RAW_TRAIN_IMAGES_PATH + key)
        im_bbox = cv2.rectangle(im_rd, (value[0][0], value[0][1]),
                                (value[1][0], value[1][1]), (255, 0, 0), 2)
        cv2.imwrite(BBOX_IMAGES_PATH + key, im_bbox)
    LOGGER.debug('Created all bounding boxes for %d images', len(json_bbox.keys()))
    LOGGER.debug('Images with bounding boxes saved to %s', BBOX_IMAGES_PATH)


def write_bbox_csv():
    json_bbox = read_json_file(BBOX_JSON_PATH)
    with open(BBOX_CSV_PATH, encoding='utf8', mode='w') as file:
        file.write('img_path, xmin, ymin, xmax, ymax, label\n')
        for key, value in json_bbox.items():
            file.write(key + ',' + str(value[0][0]) + ',' + str(value[0][1]) + ',' +
                       str(value[1][0]) + ',' + str(value[1][1]) + ', table\n')


def normalise_image_size_and_bbox():
    create_dir(NORMALISED_IMAGES_PATH)
    create_dir(NORMALISED_BBOX_IMAGES_PATH)
    LOGGER.debug('Normalising image sizes to same size')
    fp = open(MIN_DIM, 'rb')
    dim = pickle.load(fp)
    LOGGER.debug('Normalising all images to %d x %d', dim, dim)
    bbox_json = read_json_file(BBOX_JSON_PATH)
    norm_bbox_json = {}
    for image in bbox_json.keys():
        img = cv2.imread(RAW_TRAIN_IMAGES_PATH + image)
        y_ = img.shape[0]
        x_ = img.shape[1]
        img = cv2.resize(img, (dim, dim))
        cv2.imwrite(NORMALISED_IMAGES_PATH + image, img)
        x_scale = dim / x_
        y_scale = dim / y_
        tc_x = int(np.round(bbox_json[image][0][0] * x_scale))
        tc_y = int(np.round(bbox_json[image][0][1] * y_scale))
        bl_x = int(np.round(bbox_json[image][1][0] * x_scale))
        bl_y = int(np.round(bbox_json[image][1][1] * y_scale))
        norm_bbox_json[image] = [tc_x, tc_y, bl_x, bl_y]

        im_bbox = cv2.rectangle(img, (tc_x, tc_y), (bl_x, bl_y), (255, 0, 0), 2)
        cv2.imwrite(NORMALISED_BBOX_IMAGES_PATH + image, im_bbox)

    write_json_to_file(norm_bbox_json, NORM_BBOX_JSON_PATH)
