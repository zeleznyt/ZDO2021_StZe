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
from skimage.filters import threshold_otsu

from podpurne_funkce import prepare_ground_true_masks, merge_masks, f1score
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

SCALE = 4
FILTR_W = 50
FILTR_H = 50
THRESHOLD = 10
KERNEL_SIZE = 3
FILTRATION_MORPHOLOGY = 3
SVM_KERNEL = 'linear'
NEIGHBORS = 5
HIDDEN_LAYERS = (4, 8, 4)
FEATURES = []
IMG_PATH = "../../Dataset/images/"
IMG_PREP_PATH = "../../Dataset/images_preprocessed/"
#IMG_NAMES = ["Original_1305_image.jpg"]
IMG_NAMES = os.listdir(IMG_PATH)
LOG_PATH = "../log/"
MODEL_PATH = "../models/"
MODELS_USED = ['svm', 'gnb', 'knn', 'mlp']

LOG = []

ann_path = "../../Dataset/annotations/annotations.json"
with open(ann_path, 'r') as file:
    data = file.read()
data = json.loads(data)

with open('features_moments.pickle', 'rb') as file:
    features_moments = pickle.load(file)


def log_start():
    LOG.clear()
    now = datetime.now().strftime("%x %X")
    setting = {'IMG_NAMES': IMG_NAMES, 'SCALE': SCALE, 'FILTR_W': FILTR_W, 'FILTR_H': FILTR_H, 'THRESHOLD': THRESHOLD,
               'KERNEL_SIZE': KERNEL_SIZE, 'FILTRATION_MORPHOLOGY': FILTRATION_MORPHOLOGY,
               'SVM_KERNEL': SVM_KERNEL, 'NEIGHBORS': NEIGHBORS, 'HIDDEN_LAYERS': HIDDEN_LAYERS, 'FEATURES' : FEATURES}
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

    conv_img = resize(conv_img, (y_o , x_o), anti_aliasing=True)
    normalized_img = gray_img - conv_img
    normalized_img = (normalized_img + 1) / 2  # 0..1

    log_info('normalization')
    return (normalized_img, conv_img)


def thresholding(normalized_img):
    if(THRESHOLD > 0):
        # prahování, ponechat THRESHOLD/100 % nejtmavsich
        hist, bins_center = exposure.histogram(normalized_img)
        hist_sum = np.sum(hist)

        for i in range(len(hist)):
            start_hist_sum = np.sum(hist[0:i])
            if (start_hist_sum / hist_sum > THRESHOLD / 100):
                val = i - 1
                break
        mask_img = normalized_img * 255 < val
        
    else:
        val = threshold_otsu(normalized_img*255)
        mask_img = normalized_img * 255 < val


    log_info('thresholding')
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
    labels_b_f = skimage.measure.label(mask_img, background=0)
    labels = skimage.measure.label(filtered_img, background=0)

    log_info('labeling')
    log_info('objects: {} -> {}'.format(len(np.unique(labels_b_f)) - 1, len(np.unique(labels)) - 1))
    return labels


def separate_true_objects(mask_img, gt_mask):
    background_img = np.multiply((1 - gt_mask), mask_img)
    # objects_img = np.multiply( gt_mask, mask_img )
    log_info('separate_true_objects')
    return background_img  # (background_img, objects_img)


def feature_extraction(labeled_img, original_img, selected_features=[]):
    # extrakce priznaku
    props = skimage.measure.regionprops(labeled_img)

    features = []
    all = False
    if not selected_features:
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
            color_r = np.mean(color_img[:, :, 0])/255
            color_g = np.mean(color_img[:, :, 1])/255
            color_b = np.mean(color_img[:, :, 2])/255
            gray = np.mean(rgb2gray(color_img/255))

            obj_f.append((color_r-features_moments['color_r'][0])/features_moments['color_r'][1])
            obj_f.append((color_g-features_moments['color_g'][0])/features_moments['color_g'][1])
            obj_f.append((color_b-features_moments['color_b'][0])/features_moments['color_b'][1])
            obj_f.append((gray-features_moments['gray'][0])/features_moments['gray'][1])

        if ('centroid' in selected_features or all):
            xc = object_prop.local_centroid[0]
            yc = object_prop.local_centroid[1]

            obj_f.append((xc-features_moments['xc'][0])/features_moments['xc'][1])
            obj_f.append((yc-features_moments['yc'][0])/features_moments['yc'][1])

        if ('compact' in selected_features or all):
            compact = (object_prop.perimeter ** 2) / object_prop.area

            obj_f.append((compact-features_moments['compact'][0])/features_moments['compact'][1])

        if ('max_len' in selected_features or all):
            max_len = object_prop.major_axis_length

            obj_f.append((max_len-features_moments['max_len'][0])/features_moments['max_len'][1])

        if ('convex' in selected_features or all):
            convex = (object_prop.area)/object_prop.convex_area
            obj_f.append((convex-features_moments['convex'][0])/features_moments['convex'][1])

        features.append(obj_f)

    log_info('feature_extraction')
    s = ' '
    if all:
        log_info(s.join(selected_features))
    else:
        log_info('all')
    return features


# def preprocess_gray_image(gray_img, name, original_img,  forced=False):
def preprocess_gray_image(gray_img, name,  forced=False):

    if not os.path.exists(os.path.join(IMG_PREP_PATH, name+'.pickle')):
        normalized_img, conv_img = normalization(gray_img)
        mask_img = thresholding(normalized_img)
        filtered_img = filtration(mask_img)
        labeled_img = labeling(mask_img, filtered_img)
        # features_img = feature_extraction(labeled_img, original_img)
        # return (normalized_img, conv_img, mask_img, filtered_img, labeled_img, features_img)
        return (normalized_img, conv_img, mask_img, filtered_img, labeled_img)
    else:
        if not forced:
            with open(os.path.join(IMG_PREP_PATH, name+'.pickle'), 'rb') as file:
                pickled_data = pickle.load(file)
            # [normalized_img, conv_img, mask_img, filtered_img, labeled_img, features_img] = (pickle.loads(pickled_data))
            [normalized_img, conv_img, mask_img, filtered_img, labeled_img] = (pickle.loads(pickled_data))
            log_info('preprocessed image loaded: {}'.format(name))

        else:
            normalized_img, conv_img = normalization(gray_img)
            mask_img = thresholding(normalized_img)
            filtered_img = filtration(mask_img)
            labeled_img = labeling(mask_img, filtered_img)
            # features_img = feature_extraction(labeled_img, original_img)

            # pickled_data = pickle.dumps([normalized_img, conv_img, mask_img, filtered_img, labeled_img, features_img])
            pickled_data = pickle.dumps([normalized_img, conv_img, mask_img, filtered_img, labeled_img])
            with open(os.path.join(IMG_PREP_PATH, name+'.pickle'), 'wb') as file:
                pickle.dump(pickled_data, file)
            log_info('preprocessed image saved: {}'.format(name))

    # return (normalized_img, conv_img, mask_img, filtered_img, labeled_img, features_img)
    return (normalized_img, conv_img, mask_img, filtered_img, labeled_img)


def visualize_prediction(gt_mask, predicted_mask, name):
    now = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    plt.figure()

    visualized_result = np.zeros((np.size(gt_mask, 0), np.size(gt_mask, 1), 3))
    visualized_result[:, :, 0] = predicted_mask*(1-np.logical_and(gt_mask, predicted_mask).astype(int))
    visualized_result[:, :, 1] = np.logical_and(gt_mask, predicted_mask).astype(int)
    visualized_result[:, :, 2] = gt_mask*(1-np.logical_and(gt_mask, predicted_mask).astype(int))
    plt.imshow(visualized_result)
    plt.savefig(os.path.join(LOG_PATH, name + '_predicted_mask_img_' + now + '.png'))


def predict(models, name):


    original_img, gray_img = load_img(os.path.join(IMG_PATH, name))
    # (normalized_img, conv_img, mask_img, filtered_img, labeled_img, features_img) = preprocess_gray_image(gray_img, name, original_img, True)
    (normalized_img, conv_img, mask_img, filtered_img, labeled_img) = preprocess_gray_image(gray_img, name)
    features_img = feature_extraction(labeled_img, original_img, FEATURES)

    masks = []
    model_names = ['svm', 'gnb', 'knn', 'mlp']
    for ind, model in enumerate(models):
        model_name = model_names[ind]
        prediction = model.predict(features_img)
        mask = get_masks_from_predictions(labeled_img, prediction)
        log_info('Model {}, Objects detected : {}'.format(model_name, int(np.sum(prediction))))
        masks.append((model_name, mask))


        #visualize_prediction(gt_mask, mask, name+'_'+model_names[i])
        visualze_detected_objects(labeled_img, original_img, prediction, name=model_name + '_' + name)





    '''
    images = [original_img, gray_img, conv_img, normalized_img, mask_img, filtered_img, labeled_img]
    labels = ['original', 'gray', 'convolution', 'normalized', 'threshold', 'filtered', 'labeled']
    log_save_imgs(images, labels, LOG_PATH)
    '''

    return masks


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
        log_info("{}".format(name))
        gt_mask = merge_masks(m)
        gt_mask = skimage.transform.rotate(gt_mask, -90, resize=True)
        mask_objects_img = gt_mask

        original_img, gray_img = load_img(os.path.join(IMG_PATH, name))
        (normalized_img, conv_img, mask_background_img, _, _) = preprocess_gray_image(gray_img, name)
        # normalized_img, conv_img = normalization(gray_img)
        # mask_background_img = thresholding(normalized_img)
        background_img = separate_true_objects(mask_background_img, gt_mask)
        filtered_img = filtration(background_img)

        labeled_background_img = labeling(mask_background_img, filtered_img, )
        labeled_objects_img = labeling(mask_background_img, mask_objects_img)
        visualze_detected_objects(labeled_objects_img, original_img, [], name='gt_' + name)

        features_bg = features_bg + feature_extraction(labeled_background_img, original_img, FEATURES)
        features_ob = features_ob + feature_extraction(labeled_objects_img, original_img, FEATURES)
        print(features_bg[0])
        print(features_ob[0])


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
        now = datetime.now().strftime("_%d_%m_%y_%H_%M_%S")
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


def load_models_from_path(path):
    models = {}
    for model_name in os.listdir(path):
        models[model_name.replace('.pickle', '')] = (pickle.loads(load_model(os.path.join(path, model_name))))
    if set(models.keys()) != set(MODELS_USED):
        log_info('MODELS LOADED DOES NOT MATCH WITH MODELS_USED')
        return list(models.values())
    else:
        return models['svm'], models['gnb'], models['knn'], models['mlp']


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
    if(N > 20):
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


def get_masks_from_predictions(labeled_img, labels):
    N = int(np.sum(labels))
    if(N == 0):
        y, x = labeled_img.shape
        mask = np.zeros([y, x])
        return mask

    y, x = labeled_img.shape
    mask = np.zeros([y,x])

    for i in range(1, len(np.unique(labeled_img))):
        if ( labels[i-1] == 1 ):
            mask = np.add(mask, (labeled_img == i) * 1)

    return mask


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


def evaluate(predicted_mask, image_name):
    gt = prepare_ground_true_masks(data, image_name)
    if type(gt) == type(0):
        mm_gt = zeros(predicted_mask.shape)
    else:
        mm_gt = merge_masks(gt)
        mm_gt = skimage.transform.rotate(mm_gt, -90, resize=True)

    mm_pr = merge_masks(predicted_mask)
    sco = f1score(mm_gt, mm_pr)
    scb = f1score(1-mm_gt, 1-mm_pr)
    sc = (sco+scb)/2

    return sc


if __name__ == '__main__':
    #split dataset
    train_names, validation_names = split_dataset(data)


    #start training
    IMG_NAMES = train_names
    #IMG_NAMES = ["Original_1305_image.jpg"]
    (svm, gnb, knn, mlp) = train(IMG_NAMES)
    #(svm, gnb, knn, mlp) = load_models_from_path(MODEL_PATH)


    model_names = ['svm', 'gnb', 'knn', 'mlp']
    models = [svm, gnb, knn, mlp]

    #predict on training set
    log_start()
    log_info('START PREDICT')

    training_score = dict.fromkeys(model_names, 0)
    for name in IMG_NAMES:
        masks = predict(models, name)
        for m in range(len(masks)):
            model_name = masks[m][0]
            mask = masks[m][1]
            score = evaluate(mask, name)
            training_score[model_name] += score
            log_info('Image {}, Model {}, F1 score {}'.format(name, model_name, score))
        break
    for k in training_score:
        f1 = training_score[k]/len(IMG_NAMES)
        log_info('Training set, model {}, F1 score {}'.format(k, f1))

    log_info('END PREDICT')
    log_save(LOG_PATH)


    #predict on validation set
    IMG_NAMES = validation_names
    log_start()
    log_info('START PREDICT Validation set')

    validation_score = dict.fromkeys(model_names, 0)
    for name in IMG_NAMES:
        masks = predict(models, name)
        for m in range(len(masks)):
            model_name = masks[m][0]
            mask = masks[m][1]
            score = evaluate(mask, name)
            validation_score[model_name] += score
            log_info('Image {}, Model {}, F1 score {}:'.format(name, model_name, score))
            print('Image {}, Model {}, F1 score {}:'.format(name, model_name, score))

    for k in validation_score:
        f1 = validation_score[k]/len(IMG_NAMES)
        log_info('Validation set, model {}, F1 score {}:'.format(k, f1))

    log_info('END PREDICT')
    log_save(LOG_PATH)
