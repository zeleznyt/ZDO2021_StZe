from zdo2021 import main
from zdo2021 import preprocess
from zdo2021 import train
import skimage.io


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

train_names, val_names = main.split_dataset(ANNOTATIONS)
(train_names, val_names) = main.remove(train_names, val_names)

# preprocessing object
prep = preprocess.Preprocess(SCALE, FILTR_W, FILTR_H, THRESHOLD, FILTRATION_MORPHOLOGY, FEATURE_MOMENTS_PATH)

# validation  models
val = train.Train(val_names, IMG_PATH, ANNOTATIONS)
val.save_proposed_regions(prep, 'log/val')

# train models
tr = train.Train(train_names, IMG_PATH, ANNOTATIONS)
tr.save_proposed_regions(prep, 'log/train')