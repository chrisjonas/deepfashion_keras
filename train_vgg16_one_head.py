import shutil
import os
import re
import cv2
import numpy as np
from six.moves import range
import json

from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

root_dir = '/data/'

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape = (200,200,3))

# Freeze all layers for now except last convolutional block  
for layer in vgg16.layers[:15]:
    layer.trainable = False

x = Flatten()(vgg16.output)
x = Dense(4096, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dense(4096, activation='relu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)

final_model = Model(inputs=vgg16.input,
                    #outputs=[y, bbox])
                    outputs=y)

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          #'bbox': 'mean_squared_error'
                         },
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             #'bbox': ['mse']
                            })
                            
data_dict_dir = root_dir + 'data/img_dicts/'

with open(data_dict_dir + 'dict_train.json', 'r') as train_f:
    dict_train = json.load(train_f)

with open(data_dict_dir + 'dict_val.json', 'r') as val_f:
    dict_val = json.load(val_f)

with open(data_dict_dir + 'dict_test.json', 'r') as test_f:
    dict_test = json.load(test_f)
    
train_dir = root_dir + "data/img/train"
val_dir = root_dir + "data/img/val"

train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(200, 200),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(200, 200),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
final_model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=2
)