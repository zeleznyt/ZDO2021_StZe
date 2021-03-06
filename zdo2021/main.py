import json
import os
from os import path
import numpy as np
import pickle
import skimage.io

from . import preprocess
from . import podpurne_funkce

MODEL_PATH = path.join(path.dirname(__file__), '../models/mlp1.pkl')
FEATURE_MOMENTS_PATH = path.join(path.dirname(__file__), 'features_moments.pickle')

SCALE = 2
FILTR_W = 75
FILTR_H = 75
THRESHOLD = 10
FILTRATION_MORPHOLOGY = 3
FEATURES = ['rgb', 'centroid', 'compact', 'convex', 'cnn']

class VarroaDetector():
    def __init__(self):
        self.MODEL = self.load_model(MODEL_PATH)
        self.prep_obj = preprocess.Preprocess(SCALE, FILTR_W, FILTR_H, THRESHOLD, FILTRATION_MORPHOLOGY, FEATURE_MOMENTS_PATH)

    def load_model(self, path):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model

    def predict(self, data):
        """
        :param data: np.ndarray with shape [pocet_obrazku, vyska, sirka, barevne_kanaly]
        :return: shape [pocet_obrazku, vyska, sirka], 0 - nic, 1 - varroa destructor
        """
        output = np.zeros(data.shape[0:3])
        for i in range(len(output)):
            original_img, gray_img = self.prep_obj.img_to_gray(data[i])
            normalized_img, conv_img = self.prep_obj.normalization(gray_img)
            mask_img = self.prep_obj.thresholding(normalized_img)
            filtered_img = self.prep_obj.filtration(mask_img)
            labeled_img = self.prep_obj.labeling(mask_img, filtered_img)
            img_features = self.prep_obj.get_features_from_image(labeled_img, original_img, FEATURES)
            if(len(img_features) > 0):
                prediction = self.MODEL.predict(img_features)
                mask = self.get_masks_from_predictions(labeled_img, prediction)
                masks = podpurne_funkce.merge_masks(mask)
            else:
                masks = np.zeros_like(gray_img)
            output[i] = masks
        return output

    def extract_features(self, preprocess_obj, features, image_path, image_name):
        print('Prredict; Extract features for image: {}'.format(image_name))
        original_img, gray_img = preprocess_obj.load_img(os.path.join(image_path, image_name))
        normalized_img, conv_img = preprocess_obj.normalization(gray_img)
        mask_img = preprocess_obj.thresholding(normalized_img)
        filtered_img = preprocess_obj.filtration(mask_img)
        labeled_img = preprocess_obj.labeling(mask_img, filtered_img)

        return (labeled_img, preprocess_obj.get_features_from_image(labeled_img, original_img, features) )

    def models_prdict(self, models, image_path, image_names, features, preprocess_obj, annotations):
        models_results = [0] * len(models)
        for name in image_names:
            (labeled_img, img_fatures) = self.extract_features(preprocess_obj, features, image_path, name)
            for ind, model in enumerate(models):
                prediction = model.predict(img_fatures)
                mask = self.get_masks_from_predictions(labeled_img, prediction)
                score = self.evaluate(mask, name, annotations)
                models_results[ind] = models_results[ind] + score

        for i in range(len(models)):
            models_results[i] = models_results[i]/len(image_names)

        return models_results

    def get_masks_from_predictions(self, labeled_img, labels):
        N = int(np.sum(labels))
        if (N == 0):
            y, x = labeled_img.shape
            mask = np.zeros([y, x])
            return mask

        y, x = labeled_img.shape
        mask = np.zeros([y, x])

        for i in range(1, len(np.unique(labeled_img))):
            if (labels[i - 1] == 1):
                mask = np.add(mask, (labeled_img == i) * 1)

        return mask

    def evaluate(self, predicted_mask, image_name, annotations):
        gt = podpurne_funkce.prepare_ground_true_masks(annotations, image_name)
        if type(gt) == type(0):
            mm_gt = np.zeros(predicted_mask.shape)
        else:
            mm_gt = podpurne_funkce.merge_masks(gt)
            mm_gt = skimage.transform.rotate(mm_gt, -90, resize=True)

        mm_pr = podpurne_funkce.merge_masks(predicted_mask)
        sco = podpurne_funkce.f1score(mm_gt, mm_pr)
        scb = podpurne_funkce.f1score(1 - mm_gt, 1 - mm_pr)
        sc = (sco + scb) / 2

        return sc

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

def remove(train_names, val_names):

    remove = ['Original_1298_image.jpg', 'Original_1299_image.jpg', 'Original_1300_image.jpg',
              'Original_1301_image.jpg', 'Original_1303_image.jpg', 'Original_1304_image.jpg', ]
    for r in remove:
        if r in train_names:
            train_names.remove(r)
        if r in val_names:
            val_names.remove(r)

    return (train_names, val_names)

def save_model(path, clf):
    with open(path, 'wb') as file:
        pickle.dump(clf, file)

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        ann = file.read()
    ann = json.loads(ann)
    return ann