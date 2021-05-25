import pytest
import os
import skimage.io
import glob
import numpy as np
from pathlib import Path
import zdo2021.main
import skimage.measure
from skimage.draw import polygon

# cd ZDO2021
# python -m pytest

def test_run_random():
    vdd = zdo2021.main.VarroaDetector()

    # Nastavte si v operačním systém proměnnou prostředí 'VARROA_DATA_PATH' s cestou k datasetu.
    # Pokud není nastavena, využívá se testovací dataset tests/test_dataset
    dataset_path = os.getenv('VARROA_DATA_PATH_', default=Path(__file__).parent / 'test_dataset/')

    # print(f'dataset_path = {dataset_path}')
    files = glob.glob(f'{dataset_path}/images/*.jpg')
    cislo_obrazku = np.random.randint(0, len(files))
    filename = files[cislo_obrazku]

    im = skimage.io.imread(filename)
    imgs = np.expand_dims(im, axis=0)
    # print(f"imgs.shape={imgs.shape}")
    prediction = vdd.predict(imgs)


    assert prediction.shape[0] == imgs.shape[0]


    # Toto se bude spouštět všude mimo GitHub
    if not os.getenv('CI'):
        import matplotlib.pyplot as plt
        plt.imshow(prediction[0])
        plt.show()


    import json
    with open(Path(dataset_path)/"annotations/instances_default.json", 'r') as file:
        ann = file.read()

    gt_ann = json.loads(ann)
    name = filename.split('/')

    ground_true_masks = prepare_ground_true_masks(gt_ann, name[-1])
    ground_true_masks = merge_masks(ground_true_masks)

    assert f1score(ground_true_masks, prediction[0]) > 0.55


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

