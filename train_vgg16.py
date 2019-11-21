import shutil
import os
import re
import cv2
import numpy as np
from six.moves import range
import json

from keras.models import Model
from keras.layers import Dense, Flatten, Input
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

class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes: dict = None, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = preprocess_input(x)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']],
                    dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y
            
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape = (200,200,3))

# Freeze all layers for now except last convolutional block  
for layer in vgg16.layers[:15]:
    layer.trainable = False

x = Flatten()(vgg16.output)
x = Dense(4096, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dense(4096, activation='relu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)

x_bbox = Flatten()(vgg16.output)
x_bbox = Dense(4096, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
x_bbox = Dense(4096, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

final_model = Model(inputs=vgg16.input,
                    outputs=[y, bbox])

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          'bbox': 'mean_squared_error'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             'bbox': ['mse']})



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

train_iterator = DirectoryIteratorWithBoundingBoxes(train_dir, train_datagen, bounding_boxes=dict_train, target_size=(200, 200))
val_iterator = DirectoryIteratorWithBoundingBoxes(val_dir, val_datagen, bounding_boxes=dict_val,target_size=(200, 200))

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir=root_dir+'logs')
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
performance_log = CSVLogger(root_dir+'perf_log/vgg16_epoch_log.csv',separator=',',append=False)
checkpoint = ModelCheckpoint(root_dir+'models/vgg16_10_epoch.h5'
                            ,save_freq='epoch',load_weights_on_restart=True)

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)
        
final_model.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch= 6538, # 209222 train records / batch size of 32
                          epochs=3, 
                          validation_data=custom_generator(val_iterator),
                          validation_steps= 1250, # 40000 val records/ batch size of 32
                          verbose=2,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard, performance_log],
                          workers=12
                         ,use_multiprocessing=True)


for layer in final_model.layers[:-6]:
    embed_model.add(layer)
    
embed_model.add(final_model.layers[-5])

embed_model.save_weights(root_dir + 'models/vgg16_100_epoch_embed.h5')

print('training complete!')