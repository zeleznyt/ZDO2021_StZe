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



class Train():
    def __init__(self, image_names, image_path, annotations_path):
        self.features_bg = []
        self.features_ob = []
        self.IMAGE_NAMES = image_names
        self.IMAGE_PATH = image_path

        with open(annotations_path, 'r') as file:
            self.ANNOTATIONS = file.read()
        self.ANNOTATIONS = json.loads(self.ANNOTATIONS)




    def fit_models(self, models):
        print('start model fit')
        X_bg = self.features_bg
        y_bg = list(np.zeros(len(self.features_bg)))

        X_ob = self.features_ob
        y_ob = list(np.ones(len(self.features_ob)))

        X = X_bg + X_ob
        y = y_bg + y_ob
        print('data prepared, X {}, y {}'.format(len(X), len(y)))
        for model in models:
            model.fit(X, y)


        return models



    def extract_features(self, preprocess_obj, features):
        for name in self.IMAGE_NAMES:
            print('Train; Extract features for image: {}'.format(name))
            m = prepare_ground_true_masks(self.ANNOTATIONS, name)
            if type(m) == type(0):
                print('extract features: skip image {}'.format(name))
                continue

            gt_mask = merge_masks(m)
            gt_mask = skimage.transform.rotate(gt_mask, -90, resize=True)
            mask_objects_img = gt_mask

            original_img, gray_img = preprocess_obj.load_img(os.path.join(self.IMAGE_PATH, name))
            normalized_img, conv_img = preprocess_obj.normalization(gray_img)
            mask_background_img = preprocess_obj.thresholding(normalized_img)
            background_img = preprocess_obj.separate_true_objects(mask_background_img, gt_mask)
            filtered_img = preprocess_obj.filtration(background_img)

            labeled_background_img = preprocess_obj.labeling(mask_background_img, filtered_img, )
            labeled_objects_img = preprocess_obj.labeling(mask_background_img, mask_objects_img)
            #visualze_detected_objects(labeled_objects_img, original_img, [], name='gt_' + name)

            self.features_bg = self.features_bg + preprocess_obj.get_features_from_image(labeled_background_img, original_img, features)
            self.features_ob = self.features_ob + preprocess_obj.get_features_from_image(labeled_objects_img, original_img, features)


    def save_proposed_regions(self, preprocess_obj, folder):

        ind = 0
        reg_bg = []
        reg_ob = []

        for name in self.IMAGE_NAMES:
            print('Train; Extract regions for image: {}'.format(name))
            m = prepare_ground_true_masks(self.ANNOTATIONS, name)
            if type(m) == type(0):
                print('extract features: skip image {}'.format(name))
                continue

            gt_mask = merge_masks(m)
            gt_mask = skimage.transform.rotate(gt_mask, -90, resize=True)
            mask_objects_img = gt_mask

            original_img, gray_img = preprocess_obj.load_img(os.path.join(self.IMAGE_PATH, name))
            normalized_img, conv_img = preprocess_obj.normalization(gray_img)
            mask_background_img = preprocess_obj.thresholding(normalized_img)
            background_img = preprocess_obj.separate_true_objects(mask_background_img, gt_mask)
            filtered_img = preprocess_obj.filtration(background_img)

            labeled_background_img = preprocess_obj.labeling(mask_background_img, filtered_img, )
            labeled_objects_img = preprocess_obj.labeling(mask_background_img, mask_objects_img)
            # visualze_detected_objects(labeled_objects_img, original_img, [], name='gt_' + name)

            reg_bg = reg_bg + preprocess_obj.get_proposed_image_regions(labeled_background_img, original_img)
            reg_ob = reg_ob + preprocess_obj.get_proposed_image_regions(labeled_objects_img, original_img)


        if(len(reg_bg) > 0):
            with open(folder + '/bg/{}.pkl'.format(str(ind).zfill(3)), 'wb') as f:
                pickle.dump(reg_bg, f)
                ind = ind + 1

        if (len(reg_ob) > 0):
            with open(folder + '/ob/{}.pkl'.format(str(ind).zfill(3)), 'wb') as f:
                pickle.dump(reg_ob, f)
                ind = ind + 1

