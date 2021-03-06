import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import polygon
import skimage.io


def f1score(gt_ann, prediction):
    if prediction.shape[0] != gt_ann.shape[0]:
        gt_ann = skimage.transform.rotate(gt_ann, -90, resize=True)

    sco = f1class(gt_ann, prediction)
    scb = f1class(1 - gt_ann, 1 - prediction)
    sc = (sco + scb) / 2

    return sc

def f1class(gt_ann, prediction):
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

def visualize(original, mask, path):
    labeled = skimage.measure.label(mask, background=0)
    props = skimage.measure.regionprops(labeled)

    N = len(props)
    R = 1
    max_figs = 20

    if(N < 1):
        return

    if(N > max_figs):
        R = np.ceil(N/max_figs)
        N = 20

    plt.figure(figsize=(4 * N, 6 * R))

    k = 1
    for i in range(1, len(np.unique(labeled))):

        if(props[i-1]):
            object_prop = props[i - 1]
            bb = object_prop.bbox
            object_mask = (labeled == i) * 1
            object_mask = object_mask[bb[0]:bb[2], bb[1]:bb[3]]
            y, x = object_mask.shape
            object_mask_3 = np.repeat(object_mask.reshape(y, x, 1), 3, axis=2)
            color_img = original[bb[0]:bb[2], bb[1]:bb[3]] * object_mask_3

            plt.subplot(R, N, k)
            plt.imshow(color_img)
            k += 1

        plt.savefig(path)

def visualize_prediction(gt_mask, predicted_mask, original, name):

    visualized_result = np.zeros((np.size(gt_mask, 0), np.size(gt_mask, 1), 3))
    visualized_result[:,:,0] = predicted_mask*(1-np.logical_and(gt_mask, predicted_mask).astype(int))
    visualized_result[:,:,1] = np.logical_and(gt_mask, predicted_mask).astype(int)
    visualized_result[:,:,2] = gt_mask*(1-np.logical_and(gt_mask, predicted_mask).astype(int))

    visualized_result = visualized_result * 255
    for i in range(np.size(gt_mask, 0)):
        for j in range(np.size(gt_mask, 1)):
            if visualized_result[i,j,0] == 0 and visualized_result[i,j,1] == 0 and visualized_result[i,j, 2] == 0:
                visualized_result[i,j,:] = original[i,j,:]

    skimage.io.imsave(name + '_predicted_mask_img_.jpg', visualized_result)