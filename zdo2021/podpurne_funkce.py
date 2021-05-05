import numpy as np
from matplotlib import pyplot as plt

import json
from skimage.draw import polygon

import skimage.io
from matplotlib import pyplot as plt
import scipy.signal
import numpy as np
from skimage import filters, exposure


def f1score(gt_ann, prediction):
    """
    Parameters
    ----------
    gt_ann : 2d array
    prediction : 2d array
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    obj = 1
    bg = 0

    for i in range(gt_ann.shape[0]):
        for j in range(gt_ann.shape[1]):
            t = gt_ann[i, j]
            p = prediction[i, j]
            if (t == p and t == obj):
                tp = tp + 1
            elif (t == p and t == bg):
                tn = tn + 1
            elif (t == obj and p == bg):
                fn = fn + 1
            elif (t == bg and p == obj):
                fp = fp + 1

    if (tp == 0):
        return 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * ((precision * recall) / (precision + recall))

    # print("TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))
    # print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, F1))
    return F1


def prepare_ground_true_masks(gt_ann, filname):
    # get image id, shape
    im_id = -1
    height = 0
    width = 0
    for im in gt_ann['images']:
        if (im['file_name'] == filname):
            im_id = im['id']
            height = im['height']
            width = im['width']
            break

    if (im_id == -1):
        # raise Exception('No image with name {}'.format(filname))
        print('No image with name {}'.format(filname))
        return 0

    # get image annotations
    im_annotations = []
    for ann in gt_ann['annotations']:
        if (ann['image_id'] == im_id):
            im_annotations.append(ann)

    if (len(im_annotations) == 0):
        # raise Exception('No annotations for image with name {}'.format(filname))
        print('No annotations for image with name {}'.format(filname))
        masks = np.zeros((height, width, 1))
        return masks

    # mask for every object
    # bg = 0, obj = 1
    masks = np.zeros((height, width, len(im_annotations)))

    for i, ann in enumerate(im_annotations):
        seg = ann['segmentation']
        c = seg[0][0::2]
        r = seg[0][1::2]

        rr, cc = polygon(r, c)
        masks[rr, cc, i] = 1

    return masks


def merge_masks(masks):
    if len(masks.shape) < 3:
        return masks
    MASK = np.zeros(masks[:, :, 0].shape)

    for i in range(masks.shape[2]):
        MASK = np.add(MASK, masks[:, :, i])
    return MASK