import torch
import pickle
from utilities import LOGGER
from utilities.rcnn_functions import calculate_iou
from utilities.config import VGG_SCALE_SIZE, ASPECT_RATIOS, SCALES, MIN_DIM


class AnchorBox:
    def __init__(self):
        fp = open(MIN_DIM, 'rb')
        self.img_size = pickle.load(fp)
        self.anchor_boxes = self.__initialize_anchor_boxes()

    def __initialize_anchor_boxes(self):
        """
        :return:
        """
        LOGGER.debug('Entered')

        anchor_boxes = None
        vgg_output_dim = self.img_size // VGG_SCALE_SIZE
        count = 0
        for pixel_x in range(vgg_output_dim):
            for pixel_y in range(vgg_output_dim):
                for siz in SCALES:
                    scaled_size = vgg_output_dim * siz
                    for ratio in ASPECT_RATIOS:
                        h_bbox = scaled_size // ratio[0]
                        w_bbox = scaled_size // ratio[1]
                        anchor_box = torch.tensor([[pixel_x, pixel_y, w_bbox, h_bbox]])
                        if count == 0:
                            anchor_boxes = anchor_box
                        else:
                            anchor_boxes = torch.cat((anchor_boxes, anchor_box))
                        count = count + 1
        LOGGER.debug('Total Anchor Boxes = %s', str(len(anchor_box)))
        return anchor_boxes.float()

    @staticmethod
    def calculate_p_for_each_anchor_box(anchor_boxes_list, ground_truths,
                                        iou_threshold=0.7, background_iou_threshold=0.3):
        """
        This is for the classification network of the region proposal network
            this gives whether to consider an anchor box or not i.e 0,1
        :param anchor_boxes_list: requires the anchor box to bet
        :param ground_truths: this is a list of ground truth box [[class, x,y,w,h], [class, x,y,w,h]]
        :param iou_threshold:
        :param background_iou_threshold:
        :return:
            number of bounding boxes, 2
        """
        LOGGER.debug('Entered')

        li_background_index = []
        li_foreground_index = []
        reg_tensor = torch.zeros(1, 4)

        max_index = 0
        max_iou = 0
        for img_gnd_truth in ground_truths:
            gt_bbox_coor = [gt_coor // VGG_SCALE_SIZE for gt_coor in img_gnd_truth]
            count = 0
            for anchor in anchor_boxes_list:
                iou = calculate_iou(anchor.tolist(), gt_bbox_coor)
                if iou > iou_threshold:
                    reg_tensor = torch.cat((reg_tensor, torch.sub(torch.tensor(gt_bbox_coor).float(),
                                                                  anchor).unsqueeze(0)), dim=0)
                    li_foreground_index.append(count)
                elif iou < background_iou_threshold:
                    li_background_index.append(count)
                else:
                    if iou > max_iou:
                        max_index = count
                        max_iou = iou
                count = count + 1
            if len(li_foreground_index) == 0:
                li_foreground_index.append(max_index)
                reg_tensor = torch.cat((reg_tensor, torch.sub(torch.tensor(gt_bbox_coor).float(),
                                                              anchor_boxes_list[max_index].float()).unsqueeze(0)), dim=0)

            if max_index in li_background_index:
                li_background_index.remove(max_index)
        LOGGER.debug('Detected %s anchors in foreground with iou > %s',
                    str(len(li_foreground_index)), str(iou_threshold))
        LOGGER.debug('Detected %s anchors in background with iou < %s',
                    str(len(li_background_index)), str(background_iou_threshold))

        return li_foreground_index, li_background_index, reg_tensor[1:, :]
