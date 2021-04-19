import numpy as np
from matplotlib import pyplot as plt

import os
import sys
import json
from datetime import datetime
from tqdm import tqdm

import skimage.io
import scipy.signal
from skimage import filters, exposure, morphology
from skimage.color import rgb2gray

from sklearn import svm as svm_module
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from podpurne_funkce import prepare_ground_true_masks, merge_masks
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
#from . import podpurne_funkce

"""
---->TODO<----
    train_models - ukladani modelu

    load_model - nacteni modelu
        odstranit modely jako navratovou hodnotu z train_models a argument z load_model

    train - na vsech obrazcich
          - separate_true_objects pred filtration, nebo po?

    prediction - spocitat skore pres F1 - zatim jen vypis poctu detekovanych objektu

    ? pridat funkci pro nacteni modelu                            ?
    ? zrychleni - zmena velikosti pred a po normalizaci osvetleni ?
    ?           - extrakce rgb - hodně pomale                     ? 

    presun vsech funkci do VarroaDetector tridy - 
"""

#---------------------------------------------------------------------------

class VarroaDetector():
    def __init__(self):
        pass

    def predict(self, data):
        """
        :param data: np.ndarray with shape [pocet_obrazku, vyska, sirka, barevne_kanaly]
        :return: shape [pocet_obrazku, vyska, sirka], 0 - nic, 1 - varroa destructor
        """
        print("ahoj")
        output = np.zeros_like(data)
        return output



#---------------------------------------------------------------------------

FILTR_W = 50
FILTR_H = 50
THRESHOLD = 10
KERNEL_SIZE = 3
FILTRATION_MORPHOLOGY = 2
CONNECTIVITY = 2  # 1 - ctyr okoli, 2, None - osmi
IMG_PATH = "../../Dataset/images/Original_1305_image.jpg"
LOG_PATH = "../log/"

LOG = []


ann_path = "../../Dataset/annotations/annotations.json"
with open(ann_path, 'r') as file:
    data = file.read()
data = json.loads(data)




def log_start():
    LOG.clear()
    now = datetime.now().strftime("%x %X")
    setting = {'IMG_PATH': IMG_PATH, 'FILTR_W': FILTR_W, 'FILTR_H': FILTR_H, 'THRESHOLD': THRESHOLD,
               'KERNEL_SIZE': KERNEL_SIZE, 'FILTRATION_MORPHOLOGY': FILTRATION_MORPHOLOGY}
    LOG.append([now, setting])


def log_info(info):
    now = datetime.now().strftime("%x %X")
    LOG.append([now, info])
    print(now.split(' ')[1] + ' :  COMPLETED : ' + info)


def log_save(path=''):
    # LOG.append([objects_before, objects])
    now = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    with open(os.path.join(path, "log_" + now + ".json"), 'w') as f:
        json.dump(LOG, f, indent=2)


def log_save_imgs(images, labels, path=''):
    N = len(labels)

    plt.figure(figsize=(6 * N, 6))
    for n in range(N):
        plt.subplot(1, N, n + 1)
        plt.imshow(images[n], cmap='gray')
        plt.title(labels[n])

    now = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    # plt.savefig(LOG_PATH + 'result_imgs_' + now + '.png')
    plt.savefig(os.path.join(path, 'result_imgs_' + now + '.png'))


def load_img(path):
    # nacteni obrazku
    original_img = skimage.io.imread(path)  # 0..255
    gray_img = rgb2gray(original_img)  # 0..1
    log_info('load_img')
    return (original_img, gray_img)


def normalization(gray_img):
    # normalizace osvětlení
    y, x = gray_img.shape
    kernel = np.ones([int(y / FILTR_H), int(x / FILTR_W)])
    kernel = kernel / np.sum(kernel)
    conv_img = scipy.signal.convolve2d(gray_img, kernel, mode="same")  # 0..1

    normalized_img = gray_img - conv_img
    normalized_img = (normalized_img + 1) / 2  # 0..1
    log_info('normalization')
    return (normalized_img, conv_img)


def tresholding(normalized_img):
    # prahování, ponechat THRESHOLD/100 % nejtmavsich
    hist, bins_center = exposure.histogram(normalized_img)
    hist_sum = np.sum(hist)

    for i in range(len(hist)):
        start_hist_sum = np.sum(hist[0:i])
        if (start_hist_sum / hist_sum > THRESHOLD / 100):
            val = i - 1
            break

    mask_img = normalized_img * 255 < val
    log_info('tresholding')
    return mask_img


def filtration(mask_img):
    # filtrace
    kernel = np.ones([KERNEL_SIZE, KERNEL_SIZE])
    filtered_img = mask_img
    for i in range(FILTRATION_MORPHOLOGY):
        filtered_img = skimage.morphology.binary_erosion(filtered_img, kernel)
    for i in range(FILTRATION_MORPHOLOGY):
        filtered_img = skimage.morphology.binary_dilation(filtered_img, kernel)
    '''
    for o in FILTRATION_MORPHOLOGY:
      if(o == 'e'):
        filtered_img = skimage.morphology.binary_erosion(filtered_img, kernel)
      elif(o == 'd'):
        filtered_img = skimage.morphology.binary_dilation(filtered_img, kernel)
    '''
    log_info('filtration')
    return filtered_img


def labeling(mask_img, filtered_img):
    # "obarveni"
    labels_b_f = skimage.measure.label(mask_img, background=0, connectivity=CONNECTIVITY)
    labels = skimage.measure.label(filtered_img, background=0, connectivity=CONNECTIVITY)

    log_info('labeling')
    log_info('objects: {} -> {}'.format(len(np.unique(labels_b_f)) - 1, len(np.unique(labels)) - 1))
    return labels


def separate_true_objects(mask_img, gt_mask):
    background_img = np.multiply((1 - gt_mask), mask_img)
    # objects_img = np.multiply( gt_mask, mask_img )
    log_info('separate_true_objects')
    return background_img  # (background_img, objects_img)


def feture_extraction(labeled_img, original_img, selected_features=[]):
    # extrakce priznaku
    props = skimage.measure.regionprops(labeled_img)

    features = []
    all = False
    if not selected_features:
        selected_features = ['rgb', 'centroid', 'compact', 'max_len']
        all = True

    # for i in range(1, len(np.unique(labeled_img)) ):
    for i in tqdm(range(1, len(np.unique(labeled_img)))):
        object_prop = props[i - 1]
        bb = object_prop.bbox
        obj_f = []

        if ('rgb' in selected_features or all):
            object_mask = (labeled_img == i) * 1
            object_mask = object_mask[bb[0]:bb[2], bb[1]:bb[3]]
            y, x = object_mask.shape
            object_mask_3 = np.repeat(object_mask.reshape(y, x, 1), 3, axis=2)
            color_img = original_img[bb[0]:bb[2], bb[1]:bb[3]] * object_mask_3
            color_r = np.mean(color_img[:, :, 0])
            color_g = np.mean(color_img[:, :, 1])
            color_b = np.mean(color_img[:, :, 2])

            obj_f.append(color_r)
            obj_f.append(color_g)
            obj_f.append(color_b)


        if ('centroid' in selected_features or all):
            xc = object_prop.local_centroid[0]
            yc = object_prop.local_centroid[1]

            obj_f.append(xc)
            obj_f.append(yc)

        if ('compact' in selected_features or all):
            compact = (object_prop.perimeter ** 2) / object_prop.area

            obj_f.append(compact)

        if ('max_len' in selected_features or all):
            max_len = object_prop.major_axis_length

            obj_f.append(max_len)

        features.append(obj_f)


    log_info('feture_extraction')
    s = ' '
    log_info(s.join(selected_features))
    return features



def predict(models):
    log_start()
    log_info('START PREDICT')
    original_img, gray_img = load_img(IMG_PATH)
    normalized_img, conv_img = normalization(gray_img)
    mask_img = tresholding(normalized_img)
    filtered_img = filtration(mask_img)
    labeled_img = labeling(mask_img, filtered_img)

    features_img = feture_extraction(labeled_img, original_img)

    model_names = ['svm', 'gnb', 'knn', 'mlp']
    for i in range(len(models)):
        print('Objects detected {} : {}'.format(model_names[i], int(np.sum(models[i].predict(features_img)))))

    log_info('END PREDICT')
    log_save(LOG_PATH)

    images = [original_img, gray_img, conv_img, normalized_img, mask_img, filtered_img, labeled_img]
    labels = ['original', 'gray', 'convolution', 'normalized', 'treshold', 'filtered', 'labeled']
    log_save_imgs(images, labels, LOG_PATH)


def train():
    log_start()
    log_info('START TRAIN')

    m = prepare_ground_true_masks(data, 'Original_1305_image.jpg')
    gt_mask = merge_masks(m)
    gt_mask = skimage.transform.rotate(gt_mask, -90, resize=True)
    mask_objects_img = gt_mask

    original_img, gray_img = load_img(IMG_PATH)
    normalized_img, conv_img = normalization(gray_img)
    mask_background_img = tresholding(normalized_img)
    background_img = separate_true_objects(mask_background_img, gt_mask)
    filtered_img = filtration(background_img)

    labeled_background_img = labeling(mask_background_img, filtered_img)
    labeled_objects_img = labeling(mask_background_img, mask_objects_img)

    features_bg = feture_extraction(labeled_background_img, original_img)
    features_ob = feture_extraction(labeled_objects_img, original_img)

    (svm, gnb, knn, mlp) = train_models(features_bg, features_ob)

    log_info('END TRAIN')
    log_save(LOG_PATH)

    images = [original_img, gray_img, normalized_img, conv_img, background_img, mask_objects_img, mask_background_img,
              filtered_img, labeled_background_img, labeled_objects_img]
    labels = ['original_img', 'gray_img', 'normalized_img', 'conv_img', 'background_img', 'mask_objects_img',
              'mask_background_img', 'filtered_img', 'labeled_background_img', 'labeled_objects_img']
    log_save_imgs(images, labels, LOG_PATH)

    return (svm, gnb, knn, mlp)


def train_models(fe_bg, fe_ob, save=False):
    X_bg = fe_bg
    y_bg = list(np.zeros(len(fe_bg)))

    X_ob = fe_ob
    y_ob = list(np.ones(len(fe_ob)))

    X = X_bg + X_ob
    y = y_bg + y_ob

    svm = svm_module.SVC()  # kernel='linear'
    svm.fit(X, y)

    gnb = GaussianNB()
    gnb.fit(X, y)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, y)

    mlp = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(8, 4, 2), random_state=1)
    mlp.fit(X, y)

    log_info("SVM: bg = {}, ob = {}/{}".format(np.sum(svm.predict(X_bg)), int(np.sum(svm.predict(X_ob))), len(y_ob)))
    log_info("Bayes: bg = {}, ob = {}/{}".format(np.sum(gnb.predict(X_bg)), int(np.sum(gnb.predict(X_ob))), len(y_ob)))
    log_info("K-NN: bg = {}, ob = {}/{}".format(np.sum(knn.predict(X_bg)), int(np.sum(knn.predict(X_ob))), len(y_ob)))
    log_info("Multi Layer Preceptron: bg = {}, ob = {}/{}".format(np.sum(mlp.predict(X_bg)), int(np.sum(mlp.predict(X_ob))), len(y_ob)))

    if (save):
        pass

    return (svm, gnb, knn, mlp)


def load_model(path=''):
    if (path):
        pass


if __name__ == '__main__':
    (svm, gnb, knn, mlp) = train()
    predict([svm, gnb, knn, mlp])
