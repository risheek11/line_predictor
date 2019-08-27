"""
    Project: Object Detection Entry point file
    to process data, train and infer.
    Author: Goel, Ayush
    Date: 1st August 2019
"""
from data_processing.bbox import get_img_bbox_coors, create_bbox, normalise_image_size_and_bbox, \
    get_norm_xywh_format
from models.minimal import Minimal
import getopt
import sys
import torch


"""
-e   --extract     Extract the coordinates and create normalised bounding box in xywh format for json only
-t   --train       Train the model
-r   --retrain     Retrain the model
-i   --infer       Inference with model
"""

options, remainder = getopt.getopt(sys.argv[1:], shortopts='etr:i:', longopts=["extract",
                                                                               "train",
                                                                               "retrain=",
                                                                               "infer="])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for opt, arg in options:
    if opt in ('-e', '--extract'):
        get_img_bbox_coors(limit=1000)
        create_bbox()
        normalise_image_size_and_bbox()
        get_norm_xywh_format()
    if opt in ('-t', '--train'):
        minimal = Minimal()
        minimal = minimal.to(device)
        minimal.model_train()
    if opt in ('-r', '--retrain'):
        epoch_offset = int(arg.split('_')[3].split('.')[0])
        minimal = Minimal()
        minimal.load_state_dict(torch.load(arg))
        minimal = minimal.to(device)
        minimal.model_train(epoch_offset=epoch_offset)
    if opt in ('-i', '--infer'):
        path_of_state_dict = arg
        minimal = Minimal()
        minimal.load_state_dict(torch.load(arg))
        minimal = minimal.to(device)
        minimal.model_inference()
