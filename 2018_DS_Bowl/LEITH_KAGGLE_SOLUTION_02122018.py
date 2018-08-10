import os
import cv2
import sys
import random
import warnings
import numpy as np
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
 
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
 
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
 
import tensorflow as tf
 
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
 

from config import Config
import utils
import model as modellib
import visualize
from model import log
 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

import math
import re
import time
import numpy as np
import cv2
 

from config import Config
import utilsfrom subprocess import check_output
import model as modellib
import visualize
from model import log
 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import skimage.io
 
from skimage.transform import resize
 

 
from tqdm import tqdm
 
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    LEARNING_RATE = 0.001

    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    bs = 2

    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 100  

    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    TRAIN_ROIS_PER_IMAGE = 128

    MAX_GT_INSTANCES = 120

    DETECTION_MAX_INSTANCES = 120


    
config = ShapesConfig()
config.display()



 
 
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
   
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
 
#image_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/images/{}.png".format(image_id,im$
#mask_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/masks/*.png".format(image_id)

class ShapesDataset(utils.Dataset):
       
        def start(self, image_ids):
            global image_id
            i = 0
            for ax_index, image_id in enumerate(image_ids):
                self.add_class("shapes", 1, "nuclei")
                shapes = "nuclei"
                self.add_image("shapes", image_id=i, path = None, shapes = shapes) 
                print(image_id)
                print(i)
                image_id = i
                self.load_image(image_id)
                self.load_mask(image_id)
                self.image_reference(image_id)        
                i = i + 1
 
        def read_image_labels(self, image_id):
            # most of the content in this function is taken from 'Example Metric Implementation' kernel
            # by 'William Cukierski'
            image_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/images/{}.png".format($
            mask_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/masks/*.png".format(ima$
            image = skimage.io.imread(image_file)[:,:,:3]
            masks = skimage.io.imread_collection(mask_file).concatenate()   
            height, width, _ = image.shape
            num_masks = masks.shape[0]
            labels = np.zeros((height, width), np.uint16)
            for index in range(0, num_masks): 
                labels[masks[index] > 0] = index + 1
            return image, labels
           
 

        def load_image(self,image_id):
            info = self.image_info[image_id]
            image_info = self.image_info[image_id]
            image_ids = check_output(["ls", "Kaggle/stage1_train/"]).decode("utf8").split()
            image_id = image_ids[image_id]
            image, labels = self.read_image_labels(image_id)
            IMG_WIDTH = 128
            IMG_HEIGHT = 128
            IMG_CHANNELS = 3
            img = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            return img
            
 
        def load_mask(self,image_id):
            info = self.image_info[image_id]
            shapes = info['shapes']
            image_ids = check_output(["ls", "Kaggle/stage1_train/"]).decode("utf8").split()
            image_id = image_ids[image_id]
            image, labels = self.read_image_labels(image_id)
            IMG_WIDTH = 128
            IMG_HEIGHT = 128
            mask = np.expand_dims(resize(labels, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
            class_ids = []
            class_id = 1
            class_ids.append(class_id)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
 
        def image_reference(self, image_id):
            info = self.image_info[image_id]
            if info["source"] == "shapes":
                return info["shapes"]
            else:https://github.com/waleedka/coco
                super(self.__class__).image_reference(self, image_id)
 
           
 

data_train = check_output(["ls", "Kaggle/stage1_train/"]).decode("utf8").split()
image_ids_train = data_train
print(len(image_ids_train))
 
dataset_train = ShapesDataset()
dataset_train.start(image_ids_train)
dataset_train.prepare()
 

data_val = check_output(["ls", "Kaggle/stage1_train/"]).decode("utf8").split()
image_ids_val = data_val
print(len(image_ids_val))
 
dataset_val = ShapesDataset()
dataset_val.start(image_ids_val)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
 
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
 
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
 

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,

            epochs=2,
            layers="all")
 
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
inference_config = InferenceConfig()
 
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)
 
# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]
 
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
 
################################################################################
#############################  TESTING ########################################
################################################################################
 
print('Testing')
data_test = check_output(["ls", "Kaggle/stage1_test/"]).decode("utf8").split()
image_ids_test = data_test
print(len(image_ids_test))
 
dataset_test = ShapesDataset()
dataset_test.start(image_ids_test)
dataset_test.prepare()





#################################################################################
 
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
 
#################################################################################
test_path= 'Kaggle/stage1_test/'
test_ids = check_output(["ls", "Kaggle/stage1_test/"]).decode("utf8").split()
rles = []
new_test_ids = []
savedresizedimages = []
mask_origin_size_all = []
APs = [] 
image_ids = dataset_test.image_ids

x = 0
for image_id in image_ids:

    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    
    # Run object detection
    results = model.detect([image], verbose=1)
    r = results[0]    

    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)

    image_id = test_ids[x]
    x = x + 1     
    rle = list(prob_to_rles(r['masks']))

    rles.extend(rle)
    new_test_ids.extend([image_id] * len(rle))
       


sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub_02112018_v49.csv', index=False)
 
print('Win!')
print('Success')

print("mAP: ", np.mean(APs))

# Test on a random image
image_id = random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


