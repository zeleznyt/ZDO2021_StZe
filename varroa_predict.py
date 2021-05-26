from zdo2021 import podpurne_funkce
from zdo2021 import main
import os
import skimage.io
import skimage.transform
import numpy as np

HOMEPATH  = '..'
IMG_PATH = HOMEPATH  + "/Dataset/images/"
ANNOTATION_PATH = HOMEPATH  + "/Dataset/annotations/annotations.json"

ANNOTATIONS = main.load_annotations(ANNOTATION_PATH)
(train_names, val_names) = main.split_dataset(ANNOTATIONS)
(train_names, val_names) = main.remove(train_names, val_names)

vdd = main.VarroaDetector()
#files = os.listdir(IMG_PATH)
#files = val_names
#files = ['Original_1323_image.jpg']
files = train_names + val_names


F1 = []
for name in files:
    print()
    print(name)
    im = skimage.io.imread(IMG_PATH + name)
    imgs = np.expand_dims(im, axis=0)
    prediction = vdd.predict(imgs)
    #podpurne_funkce.visualize(im, prediction[0], 'log/predict/{}'.format(name))

    mask = podpurne_funkce.prepare_ground_true_masks(ANNOTATIONS, name)
    mask = podpurne_funkce.merge_masks(mask)
    mask = skimage.transform.rotate(mask, -90, resize=True)
    #podpurne_funkce.visualize(im, mask, 'log/gt/{}'.format(name))

    f1 = podpurne_funkce.f1score(mask, prediction[0])
    F1.append(f1)
    print('image: {}; F1: {}'.format(name, f1))
    #podpurne_funkce.visualize_prediction(mask, prediction[0], im, 'log/' + name)

print()
print('F1: {}'.format(np.mean(F1)))
print(F1)
print(files)