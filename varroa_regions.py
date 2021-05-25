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

train_names, validation_names = main.split_dataset(ANNOTATIONS)
remove = ['Original_1298_image.jpg', 'Original_1299_image.jpg', 'Original_1300_image.jpg', 'Original_1301_image.jpg',
          'Original_1303_image.jpg', 'Original_1304_image.jpg', ]

for r in remove:
    if r in train_names:
        train_names.remove(r)
    if r in validation_names:
        validation_names.remove(r)

# preprocessing object
prep = preprocess.Preprocess(SCALE, FILTR_W, FILTR_H, THRESHOLD, FILTRATION_MORPHOLOGY, FEATURE_MOMENTS_PATH)

# validation  models
val = train.Train(validation_names, IMG_PATH, ANNOTATIONS)
val.save_proposed_regions(prep, 'log/val')

# train models
tr = train.Train(train_names, IMG_PATH, ANNOTATIONS)
tr.save_proposed_regions(prep, 'log/train')