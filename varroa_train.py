from zdo2021 import main
from zdo2021 import preprocess
from zdo2021 import train
import skimage.io
from sklearn import svm as svm_module
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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


(train_names, val_names) = main.split_dataset(ANNOTATIONS)
(train_names, val_names) = main.remove(train_names, val_names)

# models
svm11 = svm_module.SVC(kernel='rbf', gamma=1, C=100)
knn5 = KNeighborsClassifier(n_neighbors=5)
mlp1 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(128, 256, 512), random_state=1)

models = [svm11, knn5, mlp1]
model_names = ['svm11', 'knn5', 'mlp1']

#preprocessing object
prep = preprocess.Preprocess(SCALE, FILTR_W, FILTR_H, THRESHOLD, FILTRATION_MORPHOLOGY, FEATURE_MOMENTS_PATH)

#train models
tr = train.Train(train_names + val_names, IMG_PATH, ANNOTATIONS)
tr.extract_features(prep, FEATURES)
trained_models = tr.fit_models(models)

#save models
MODEL_PATH = 'models/'
for i, m in enumerate(trained_models):
    main.save_model(MODEL_PATH + model_names[i] + '.pkl' , m)


#predict on training set
#predict = main.VarroaDetector()
#score = predict.models_prdict(trained_models, IMG_PATH, train_names, FEATURES, prep, ANNOTATIONS)
#print('Training set results')
#for i in range(len(score)):
#    print('Model: {}, F1: {}'.format(models_names[i], score[i]))

# predict on validation set
#predict = main.VarroaDetector()
#score = predict.models_prdict(trained_models, IMG_PATH, val_names, FEATURES, prep, ANNOTATIONS)
#print('Validation set results')
#for i in range(len(score)):
#    print('Model: {}, F1: {}'.format(models_names[i], score[i]))