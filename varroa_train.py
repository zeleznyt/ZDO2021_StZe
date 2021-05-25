from zdo2021 import podpurne_funkce
from zdo2021 import main
from zdo2021 import preprocess
from zdo2021 import train
import os
import skimage.io
import skimage.transform
import numpy as np
from sklearn import svm as svm_module
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

HOMEPATH  = '..'
IMG_PATH = HOMEPATH  + "/Dataset/images/"
ANNOTATION_PATH = HOMEPATH  + "/Dataset/annotations/annotations.json"
ANNOTATIONS = main.load_annotations(ANNOTATION_PATH)
FEATURE_MOMENTS_PATH = 'zdo2021/features_moments.pickle'

SCALE = 2
FILTR_W = 75
FILTR_H = 75
THRESHOLD = 10
FILTRATION_MORPHOLOGY = 3
FEATURES = ['rgb', 'centroid', 'compact', 'convex', 'cnn']


train_names, validation_names = main.split_dataset(ANNOTATIONS)
remove = ['Original_1298_image.jpg', 'Original_1299_image.jpg', 'Original_1300_image.jpg', 'Original_1301_image.jpg',
          'Original_1303_image.jpg', 'Original_1304_image.jpg', ]

for r in remove:
    if r in train_names:
        train_names.remove(r)
    if r in validation_names:
        validation_names.remove(r)

    # models
svm1 = svm_module.SVC(kernel='linear')
svm2 = svm_module.SVC(kernel='poly', degree=2)
svm6 = svm_module.SVC(kernel='rbf', gamma=0.1, C=0.01)
svm7 = svm_module.SVC(kernel='rbf', gamma=0.1, C=1)
svm8 = svm_module.SVC(kernel='rbf', gamma=0.1, C=100)
svm9 = svm_module.SVC(kernel='rbf', gamma=1, C=0.01)
svm10 = svm_module.SVC(kernel='rbf', gamma=1, C=1)
svm11 = svm_module.SVC(kernel='rbf', gamma=1, C=100)
svm12 = svm_module.SVC(kernel='rbf', gamma=10, C=0.01)
svm13 = svm_module.SVC(kernel='rbf', gamma=10, C=1)
svm14 = svm_module.SVC(kernel='rbf', gamma=10, C=100)
svm15 = svm_module.SVC(kernel='sigmoid', gamma=0.1)
svm16 = svm_module.SVC(kernel='sigmoid', gamma=1)
svm17 = svm_module.SVC(kernel='sigmoid', gamma=10)

gnb = GaussianNB()

knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)

mlp1 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(128, 256, 512), random_state=1)
mlp2 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(4, 8, 8, 16, 32), random_state=1)
mlp3 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(16, 64, 64, 128, 512, 128, 16), random_state=1)

models = [svm1, svm2, svm6, svm7, svm8, svm9, svm10, svm11, svm12, svm13, svm14, svm15, svm16, svm17,
          gnb, knn3, knn5, knn7, mlp1, mlp2, mlp3]

models_names = ['svm1', 'svm2', 'svm6', 'svm7', 'svm8', 'svm9', 'svm10', 'svm11', 'svm12', 'svm13', 'svm14', 'svm15',
                'svm16', 'svm17','gnb', 'knn3', 'knn5', 'knn7', 'mlp1', 'mlp2', 'mlp3']

#preprocessing object
prep = preprocess.Preprocess(SCALE, FILTR_W, FILTR_H, THRESHOLD, FILTRATION_MORPHOLOGY, FEATURE_MOMENTS_PATH)


#train models
tr = train.Train(train_names, IMG_PATH, ANNOTATIONS)
tr.extract_features(prep, FEATURES)
trained_models = tr.fit_models(models)


#save models
'''
for i, m in enumerate(trained_models):
    main.save_model(MODEL_PATH + models_names[i] + '.pkl' , m)
'''


#predict on training set
predict = main.VarroaDetector()
score = predict.models_prdict(trained_models, IMG_PATH, train_names, FEATURES, prep, ANNOTATIONS)
print('Training set results')
for i in range(len(score)):
    print('Model: {}, F1: {}'.format(models_names[i], score[i]))




# predict on validation set
predict = main.VarroaDetector()
score = predict.models_prdict(trained_models, IMG_PATH, validation_names, FEATURES, prep, ANNOTATIONS)
print('Validation set results')
for i in range(len(score)):
    print('Model: {}, F1: {}'.format(models_names[i], score[i]))

