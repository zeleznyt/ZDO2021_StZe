import numpy as np
from matplotlib import pyplot as plt

import os
import sys
import json
import pickle
from datetime import datetime
from tqdm import tqdm

import skimage.io
import scipy.signal
from skimage import filters, exposure, morphology
from skimage.color import rgb2gray
from skimage.transform import resize

import sklearn
from sklearn import svm as svm_module
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from podpurne_funkce import prepare_ground_true_masks, merge_masks
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
#from . import podpurne_funkce

"""
---->TODO<----

    load_model - odstranit modely jako navratovou hodnotu z train_models a argument z load_model

    train - separate_true_objects pred filtration, nebo po?

    prediction - spocitat skore pres F1 - zatim jen vypis poctu detekovanych objektu

    ? pridat funkci pro nacteni modelu                            ?
    ? zrychleni -  extrakce rgb - hodně pomale                      ?


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

SCALE = 1
FILTR_W = 50
FILTR_H = 50
THRESHOLD = 10
KERNEL_SIZE = 3
FILTRATION_MORPHOLOGY = 2
CONNECTIVITY = 2  # 1 - ctyr okoli, 2, None - osmi
SVM_KERNEL = 'linear'
NEIGHBORS = 3
HIDDEN_LAYERS = (4, 8, 4)
IMG_PATH = "../../Dataset/images/"
IMG_NAMES = ["Original_1305_image.jpg"]
#IMG_NAMES = os.listdir(IMG_PATH)
LOG_PATH = "../log/"
MODEL_PATH = "../models/"

LOG = []


ann_path = "../../Dataset/annotations/annotations.json"
with open(ann_path, 'r') as file:
    data = file.read()
data = json.loads(data)




def log_start():
    LOG.clear()
    now = datetime.now().strftime("%x %X")
    setting = {'IMG_NAMES': IMG_NAMES, 'SCALE': SCALE,'FILTR_W': FILTR_W, 'FILTR_H': FILTR_H, 'THRESHOLD': THRESHOLD,
               'KERNEL_SIZE': KERNEL_SIZE, 'FILTRATION_MORPHOLOGY': FILTRATION_MORPHOLOGY, 'CONNECTIVITY': CONNECTIVITY,
               'SVM_KERNEL': SVM_KERNEL, 'NEIGHBORS': NEIGHBORS, 'HIDDEN_LAYERS': HIDDEN_LAYERS}
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
    y_o, x_o = gray_img.shape
    gray_resized_img = resize(gray_img, (y_o // SCALE, x_o // SCALE), anti_aliasing=True)

    y, x = gray_resized_img.shape
    kernel = np.ones([int(y / FILTR_H), int(x / FILTR_W)])
    kernel = kernel / np.sum(kernel)
    conv_img = scipy.signal.convolve2d(gray_resized_img, kernel, mode="same")  # 0..1

    normalized_img = gray_resized_img - conv_img
    normalized_img = (normalized_img + 1) / 2  # 0..1
    normalized_img = resize(normalized_img, (y_o , x_o), anti_aliasing=True)
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
            '''
            object_mask = (labeled_img == i) * 1
            object_mask = object_mask[bb[0]:bb[2], bb[1]:bb[3]]
            y, x = object_mask.shape
            color_img = np.zeros([y,x,3])
            color_img[:, :, 0] = np.multiply( original_img[bb[0]:bb[2], bb[1]:bb[3], 0], object_mask)
            color_img[:, :, 1] = np.multiply( original_img[bb[0]:bb[2], bb[1]:bb[3], 1], object_mask)
            color_img[:, :, 2] = np.multiply( original_img[bb[0]:bb[2], bb[1]:bb[3], 2], object_mask)
            '''
            color_r = np.mean(color_img[:, :, 0])
            color_g = np.mean(color_img[:, :, 1])
            color_b = np.mean(color_img[:, :, 2])
            gray = np.mean(rgb2gray(color_img))

            obj_f.append(color_r)
            obj_f.append(color_g)
            obj_f.append(color_b)
            obj_f.append(gray)



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

        if ('convex' in selected_features or all):
            convex = (object_prop.area**2)/object_prop.convex_area
            obj_f.append(convex)

        features.append(obj_f)


    log_info('feture_extraction')
    s = ' '
    log_info(s.join(selected_features))
    return features



def predict(models, img_names):
    log_start()
    log_info('START PREDICT')
    model_masks = {}
    for name in img_names:
        original_img, gray_img = load_img(os.path.join(IMG_PATH, name))
        normalized_img, conv_img = normalization(gray_img)
        mask_img = tresholding(normalized_img)
        filtered_img = filtration(mask_img)
        labeled_img = labeling(mask_img, filtered_img)

        features_img = feture_extraction(labeled_img, original_img)

        model_names = ['svm', 'gnb', 'knn', 'mlp']
        model_masks[name] = []
        for i in range(len(models)):
            prediction = models[i].predict(features_img)
            visualze_detected_objects(labeled_img, original_img, prediction, name=model_names[i]+'_')
            masks = get_masks_from_predictions(labeled_img, prediction)
            model_masks[name].append((model_names[i], masks))
            log_info('Objects detected {} : {}'.format(model_names[i], int(np.sum(prediction))))

    log_info('END PREDICT')
    log_save(LOG_PATH)

    images = [original_img, gray_img, conv_img, normalized_img, mask_img, filtered_img, labeled_img]
    labels = ['original', 'gray', 'convolution', 'normalized', 'treshold', 'filtered', 'labeled']
    log_save_imgs(images, labels, LOG_PATH)
    return model_masks


def train(img_names):
    log_start()
    log_info('START TRAIN')
    features_bg = []
    features_ob = []

    for name in img_names:

        m = prepare_ground_true_masks(data, name)
        if type(m) == type(0):
            log_info("Obrazek {} vynechan.".format(name))
            continue
        gt_mask = merge_masks(m)
        gt_mask = skimage.transform.rotate(gt_mask, -90, resize=True)
        mask_objects_img = gt_mask


        original_img, gray_img = load_img(os.path.join(IMG_PATH, name))
        normalized_img, conv_img = normalization(gray_img)
        mask_background_img = tresholding(normalized_img)
        background_img = separate_true_objects(mask_background_img, gt_mask)
        filtered_img = filtration(background_img)

        labeled_background_img = labeling(mask_background_img, filtered_img, )
        labeled_objects_img = labeling(mask_background_img, mask_objects_img)

        visualze_detected_objects(labeled_objects_img, original_img, name='gt_')


        features_bg = features_bg + feture_extraction(labeled_background_img, original_img)
        features_ob = features_ob + feture_extraction(labeled_objects_img, original_img)


    (svm, gnb, knn, mlp) = train_models(features_bg, features_ob, True)

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

    svm = svm_module.SVC(kernel=SVM_KERNEL)
    svm.fit(X, y)

    gnb = GaussianNB()
    gnb.fit(X, y)

    knn = KNeighborsClassifier(n_neighbors=NEIGHBORS)
    knn.fit(X, y)

    mlp = MLPClassifier(alpha=1e-5, hidden_layer_sizes=HIDDEN_LAYERS, random_state=1)
    mlp.fit(X, y)

    log_info("SVM: bg = {}, ob = {}/{}".format(np.sum(svm.predict(X_bg)), int(np.sum(svm.predict(X_ob))), len(y_ob)))
    log_info("Bayes: bg = {}, ob = {}/{}".format(np.sum(gnb.predict(X_bg)), int(np.sum(gnb.predict(X_ob))), len(y_ob)))
    log_info("K-NN: bg = {}, ob = {}/{}".format(np.sum(knn.predict(X_bg)), int(np.sum(knn.predict(X_ob))), len(y_ob)))
    log_info("Multi Layer Preceptron: bg = {}, ob = {}/{}".format(np.sum(mlp.predict(X_bg)), int(np.sum(mlp.predict(X_ob))), len(y_ob)))

    if (save):
        save_model(os.path.join(MODEL_PATH, 'svm.pickle'), svm)
        save_model(os.path.join(MODEL_PATH, 'gnb.pickle'), gnb)
        save_model(os.path.join(MODEL_PATH, 'knn.pickle'), knn)
        save_model(os.path.join(MODEL_PATH, 'mlp.pickle'), mlp)

    return (svm, gnb, knn, mlp)


def save_model(path, clf):
    model = pickle.dumps(clf)
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)

    return model

def visualze_detected_objects(labeled_img, original_img, obj=[], name=''):
    props = skimage.measure.regionprops(labeled_img)
    props2 = []
    if(len(obj) > 0):
        for i in range(len(props)):
            if (obj[i] == 1):
                props2.append(props[i])
            else:
                props2.append([])
        props = props2
    else:
        obj.append(len(props))

    N = np.sum(obj)
    if(N < 1):
        return

    plt.figure(figsize=(6 * N, 8))
    k = 1
    for i in range(1, len(np.unique(labeled_img))):
        if(props[i-1]):
            object_prop = props[i - 1]
            bb = object_prop.bbox

            object_mask = (labeled_img == i) * 1
            object_mask = object_mask[bb[0]:bb[2], bb[1]:bb[3]]
            y, x = object_mask.shape
            object_mask_3 = np.repeat(object_mask.reshape(y, x, 1), 3, axis=2)
            color_img = original_img[bb[0]:bb[2], bb[1]:bb[3]] * object_mask_3

            plt.subplot(1, N, k)
            plt.imshow(color_img)
            k += 1

    now = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    # plt.savefig(LOG_PATH + 'result_imgs_' + now + '.png')
    plt.savefig(os.path.join(LOG_PATH, name + 'detected_objects_img_' + now + '.png'))


def get_masks_from_predictions(labeled_img, obj):
    # obj = prediction_labels
    N = int(np.sum(obj))
    y, x = labeled_img.shape
    masks = np.zeros([y,x,N])

    k = 0
    for i in range(1, len(np.unique(labeled_img))):
        if ( obj[i-1] == 1 ):
            masks[:,:,k] = (labeled_img == i) * 1
            k = k + 1
    return masks

def split_dataset(annotations):
    im = annotations['images']
    im_names = []
    im_ids = []

    for i in im:
        im_names.append(i['file_name'])
        im_ids.append(i['id'])

    #train_test_split(im_ids, test_size = 0.2)
    val_id = [23, 4, 25, 14, 30, 16]
    train_names = []
    val_names = []

    for i in range(len(im_ids)):
        if im_ids[i] in val_id:
            val_names.append(im_names[i])
        else:
            train_names.append(im_names[i])
    return (train_names, val_names)





if __name__ == '__main__':
    train_names, validation_names = split_dataset(data)
    
    (svm, gnb, knn, mlp) = train(IMG_NAMES)
    predicted_masks = predict([svm, gnb, knn, mlp], IMG_NAMES)



