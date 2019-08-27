"""
    Intersection over union
"""

def get_corners(coordinates):
    x_max = float(coordinates[0]) + float(coordinates[2] / 2)
    x_min = float(coordinates[0]) - float(coordinates[2] / 2)
    y_max = float(coordinates[1]) + float(coordinates[3] / 2)
    y_min = float(coordinates[1]) - float(coordinates[3] / 2)
    return x_max, x_min, y_max, y_min


def calculate_iou(anchor_box_coors, ground_truth_box_coors):
    '''

    :param anchor_box_coors: this is x_mp, y_mp, w, h
    :param ground_truth_box_coors: this is this is x_mp, y_mp, w, h
    :return:
    '''
    area_of_anchor_box = anchor_box_coors[2] * anchor_box_coors[3]
    area_of_ground_truth_box = ground_truth_box_coors[2] * ground_truth_box_coors[3]

    x_max_anchor, x_min_anchor, y_max_anchor, y_min_anchor = get_corners(anchor_box_coors)
    x_max_gt, x_min_gt, y_max_gt, y_min_gt = get_corners(ground_truth_box_coors)

    x_min_of_intersecting_rectangle = max(x_min_anchor, x_min_gt)
    y_min_of_intersecting_rectangle = max(y_min_anchor, y_min_gt)
    x_max_of_intersecting_rectangle = min(x_max_anchor, x_max_gt)
    y_max_of_intersecting_rectangle = min(y_max_anchor, y_max_gt)

    area_of_intersection = max(0.0,
                               x_max_of_intersecting_rectangle -
                               x_min_of_intersecting_rectangle) * max(0.0,
                                                                      y_max_of_intersecting_rectangle -
                                                                      y_min_of_intersecting_rectangle)
    iou = float(area_of_intersection) / float(area_of_anchor_box + area_of_ground_truth_box - area_of_intersection)
    return iou


def diff_between_list(l1, l2):
    if len(l1) == len(l2):
        count = 0
        li = []
        for el in l1:
            li.append(el - l2[count])
            count = count + 1
        return li
    else:
        -1
