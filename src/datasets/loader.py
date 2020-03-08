"""Dataset loader"""

import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.model_selection import train_test_split_balanced
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .ciarp import load_ciarp
from .lsa16 import load_lsa16
from .rwth import load_rwth

def load(dataset_name, datagen_flow=False,
         weight_classes=False, batch_size=32,
         rotation_range = 10, width_shift_range = 0.10,
         height_shift_range = 0.10, horizontal_flip = True,
         train_size=None, test_size=None,
         n_train_per_class=None, n_test_per_class=None):
    """
    Load specific dataset.

    Args:
        dataset_name (str): name of the dataset.

    Returns (train_gen, val_gen, test_gen, nb_classes, image_shape, class_weights):.
    """

    if dataset_name == "ciarp":
        x, y = load_ciarp(dataset_name) 
    elif dataset_name == "lsa16":
        x, y = load_lsa16(dataset_name)
    elif dataset_name == "rwth":
        x, y = load_rwth(dataset_name)
    else:
        raise ValueError("Unknow dataset: {}".format(dataset_name))

    image_shape = np.shape(x)[1:]

    split = train_test_split if n_train_per_class <= 0 else train_test_split_balanced

    if n_train_per_class <= 0:
        x_train, x_test, y_train, y_test = split(x, y, train_size=train_size, test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = split(x, y, n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)

    x_train, x_val, y_train, y_val = split(x_train, y_train, train_size=0.8, test_size=0.2)
    nb_classes = len(np.unique(y))

    class_weights = None
    if weight_classes:
        class_weights = compute_class_weight('balanced', np.unique(y), y)
    
    train_datagen_args = dict(featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=rotation_range,
                              width_shift_range=width_shift_range,
                              height_shift_range=height_shift_range,
                              horizontal_flip=horizontal_flip,
                              fill_mode='constant',
                              cval=0)
    train_datagen = ImageDataGenerator(train_datagen_args)
    train_datagen.fit(x_train)

    test_datagen_args = dict(featurewise_center=True,
                            featurewise_std_normalization=True,
                            fill_mode='constant',
                            cval=0)
    test_datagen = ImageDataGenerator(test_datagen_args)
    test_datagen.fit(x_train)

    val_datagen = ImageDataGenerator(test_datagen_args)
    val_datagen.fit(x_train)

    train = (train_datagen, train_datagen_args, len(x_train), len(y_train))
    val = (val_datagen, test_datagen_args, len(x_val), len(y_val))
    test = (test_datagen, test_datagen_args, len(x_test), len(y_test))

    if datagen_flow:
        # create data generators
        train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
        val_gen = val_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
        test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

        train = (train_gen, len(x_train), len(y_train))
        val = (val_gen, len(x_val), len(y_val))
        test = (test_gen, len(x_test), len(y_test))

    return train, val, test, nb_classes, image_shape, class_weights
