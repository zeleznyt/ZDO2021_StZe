import numpy as np
from matplotlib import pyplot as plt
from .cnn import cnn_predict

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





class Preprocess():
    def __init__(self, scale, filtr_h, filtr_w, threshold, morphology, feature_moments_path):
        self.SCALE = scale
        self.FILTR_H = filtr_h
        self.FILTR_W = filtr_w
        self.THRESHOLD = threshold
        self.FILTRATION_MORPHOLOGY = morphology
        self.MARGIN = 20

        with open(feature_moments_path, 'rb') as file:
            self.features_moments = pickle.load(file)



    def load_img(self, path):
        # nacteni obrazku
        original_img = skimage.io.imread(path)  # 0..255
        gray_img = rgb2gray(original_img)  # 0..1

        return (original_img, gray_img)

    def img_to_gray(self, data):
        # nacteni obrazku
        original_img = data  # 0..255
        gray_img = rgb2gray(original_img)  # 0..1

        return (original_img, gray_img)


    def normalization(self, gray_img):
        # normalizace osvětlení

        y_o, x_o = gray_img.shape
        gray_resized_img = resize(gray_img, (y_o // self.SCALE, x_o // self.SCALE), anti_aliasing=True)

        y, x = gray_resized_img.shape

        kernel = np.ones([int(y / self.FILTR_H), int(x / self.FILTR_W)])
        kernel = kernel / np.sum(kernel)
        conv_img = scipy.signal.convolve2d(gray_resized_img, kernel, mode="same")  # 0..1

        conv_img = resize(conv_img, (y_o , x_o), anti_aliasing=True)
        normalized_img = gray_img - conv_img
        normalized_img = (normalized_img + 1) / 2  # 0..1


        return (normalized_img, conv_img)


    def thresholding(self, normalized_img):
        if(self.THRESHOLD > 0):
            # prahování, ponechat THRESHOLD/100 % nejtmavsich
            hist, bins_center = exposure.histogram(normalized_img)
            hist_sum = np.sum(hist)

            for i in range(len(hist)):
                start_hist_sum = np.sum(hist[0:i])
                if (start_hist_sum / hist_sum > self.THRESHOLD / 100):
                    val = i - 1
                    break
            mask_img = normalized_img * 255 < val
            
        else:
            val = threshold_otsu(normalized_img*255)
            mask_img = normalized_img * 255 < val

        return mask_img


    def filtration(self, mask_img):
        # filtrace
        mask_img = mask_img.astype(bool)
        mask_img = skimage.morphology.remove_small_holes(mask_img)
        kernel = np.ones([3, 3])
        filtered_img = mask_img
        for i in range(self.FILTRATION_MORPHOLOGY):
            filtered_img = skimage.morphology.binary_erosion(filtered_img, kernel)
        for i in range(self.FILTRATION_MORPHOLOGY):
            filtered_img = skimage.morphology.binary_dilation(filtered_img, kernel)

        return filtered_img


    def labeling(self, mask_img, filtered_img):
        # "obarveni"
        labels_b_f = skimage.measure.label(mask_img, background=0)
        labels = skimage.measure.label(filtered_img, background=0)

        print('objects: {} -> {}'.format(len(np.unique(labels_b_f)) - 1, len(np.unique(labels)) - 1))
        return labels


    def separate_true_objects(self, mask_img, gt_mask):
        background_img = np.multiply((1 - gt_mask), mask_img)
        # objects_img = np.multiply( gt_mask, mask_img )
        return background_img  # (background_img, objects_img)


    def get_features_from_image(self, labeled_img, original_img, selected_features=[]):
        # extrakce priznaku
        props = skimage.measure.regionprops(labeled_img)

        features = []
        all = False
        if not selected_features:
            all = True
        if 'cnn' in selected_features:
            this_dir, this_filename = os.path.split(__file__)
            DATA_PATH = os.path.join(this_dir, "cnn", "log", "varoa_net_03.pth")
            net = cnn_predict.load_net(DATA_PATH)

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

                color_r = np.mean(color_img[:, :, 0]) / 255
                color_g = np.mean(color_img[:, :, 1]) / 255
                color_b = np.mean(color_img[:, :, 2]) / 255
                gray = np.mean(rgb2gray(color_img / 255))

                obj_f.append((color_r - self.features_moments['color_r'][0]) / self.features_moments['color_r'][1])
                obj_f.append((color_g - self.features_moments['color_g'][0]) / self.features_moments['color_g'][1])
                obj_f.append((color_b - self.features_moments['color_b'][0]) / self.features_moments['color_b'][1])
                obj_f.append((gray - self.features_moments['gray'][0]) / self.features_moments['gray'][1])

            if ('rgb_relative' in selected_features or all):
                object_mask = (labeled_img == i) * 1
                object_mask = object_mask[bb[0]:bb[2], bb[1]:bb[3]]
                y, x = object_mask.shape
                object_mask_3 = np.repeat(object_mask.reshape(y, x, 1), 3, axis=2)
                object_mask_margin = (labeled_img == i) * 1
                object_mask_margin = object_mask_margin[bb[0]-self.MARGIN:bb[2]+self.MARGIN, bb[1]-self.MARGIN:bb[3]+self.MARGIN]
                y_margin, x_margin = object_mask_margin.shape
                object_mask_3_margin = np.repeat(object_mask_margin.reshape(y_margin, x_margin, 1), 3, axis=2)
                color_img = original_img[bb[0]:bb[2], bb[1]:bb[3]] * object_mask_3
                object_margin = original_img[bb[0]-self.MARGIN:bb[2]+self.MARGIN, bb[1]-self.MARGIN:bb[3]+self.MARGIN] * object_mask_3_margin

                color_r_rel = np.mean(color_img[:, :, 0]) / np.mean(object_margin[:, :, 0])
                color_g_rel = np.mean(color_img[:, :, 1]) / np.mean(object_margin[:, :, 1])
                color_b_rel = np.mean(color_img[:, :, 2]) / np.mean(object_margin[:, :, 2])
                gray_rel = np.mean(rgb2gray(color_img)) / np.mean(object_margin)

                obj_f.append((color_r_rel - self.features_moments['color_r_rel'][0]) / self.features_moments['color_r_rel'][1])
                obj_f.append((color_g_rel - self.features_moments['color_g_rel'][0]) / self.features_moments['color_g_rel'][1])
                obj_f.append((color_b_rel - self.features_moments['color_b_rel'][0]) / self.features_moments['color_b_rel'][1])
                obj_f.append((gray_rel - self.features_moments['gray_rel'][0]) / self.features_moments['gray_rel'][1])

            if ('centroid' in selected_features or all):
                xc = object_prop.local_centroid[0]
                yc = object_prop.local_centroid[1]

                obj_f.append((xc - self.features_moments['xc'][0]) / self.features_moments['xc'][1])
                obj_f.append((yc - self.features_moments['yc'][0]) / self.features_moments['yc'][1])

            if ('compact' in selected_features or all):
                compact = (object_prop.perimeter ** 2) / object_prop.area

                obj_f.append((compact - self.features_moments['compact'][0]) / self.features_moments['compact'][1])


            if ('convex' in selected_features or all):
                convex = (object_prop.area) / object_prop.convex_area
                obj_f.append((convex - self.features_moments['convex'][0]) / self.features_moments['convex'][1])


            if('cnn' in selected_features or all ):
                color_img = original_img[bb[0]:bb[2], bb[1]:bb[3]]
                image = color_img.copy()
                out = cnn_predict.predict(net, image)
                obj_f.append(out[0])
                obj_f.append(out[1])

            if ('corners_2' in selected_features or all):
                corners_2 = len(skimage.feature.corner_peaks(skimage.feature.corner_harris(object_mask), min_distance=2))
                obj_f.append((corners_2 - self.features_moments['corners_2'][0]) / self.features_moments['corners_2'][1])

            if ('corners_3' in selected_features or all):
                corners_3 = len(skimage.feature.corner_peaks(skimage.feature.corner_harris(object_mask), min_distance=3))
                obj_f.append((corners_3 - self.features_moments['corners_3'][0]) / self.features_moments['corners_3'][1])

            features.append(obj_f)

        return features

    def get_proposed_image_regions(self, labeled_img, original_img):
        # extrakce priznaku
        props = skimage.measure.regionprops(labeled_img)
        regions = []

        # for i in range(1, len(np.unique(labeled_img)) ):
        for i in tqdm(range(1, len(np.unique(labeled_img)))):
            object_prop = props[i - 1]
            bb = object_prop.bbox
            region = original_img[bb[0]:bb[2], bb[1]:bb[3]]
            regions.append(region)

        return regions














