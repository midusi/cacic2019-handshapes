"""Dataset loader"""

import os
import numpy as np
from tensorflow.keras.datasets import cifar10, mnist
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .mnist import load_mnist
from .ciarp import load_ciarp
from .lsa16 import load_lsa16
from .rwth import load_rwth

def load(config, datagen_flow=False, with_datasets=False):
    """
    Load specific dataset.

    Args:
        dataset_name (str): name of the dataset.

    Returns (train_gen, val_gen, test_gen, nb_classes, image_shape, class_weights):.
    """

    dataset_path = '/tf/data/{}'.format(config['data.dataset'])

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # The data, split between train and test sets:
    if config['data.dataset'] == "cifar10":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(config['data.dataset'])
    elif config['data.dataset'] == "cifar100":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar100(config['data.dataset'])
    elif config['data.dataset'] == "mnist":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(config['data.dataset'])
    elif config['data.dataset'] == "Ciarp":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_ciarp(config)
    elif config['data.dataset'] == "lsa16":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_lsa16(config)
    elif config['data.dataset'] == "rwth":
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_rwth(config)
    else:
        raise ValueError("Unknow dataset: {}".format(config['data.dataset']))

    x_train, x_test, x_val = x_train / 255.0, x_test / 255.0, x_val / 255.0

    image_shape = np.shape(x_train)[1:]
    nb_classes = len(np.unique(y_train))

    class_weights = None
    if config['data.weight_classes']:
        class_weights = compute_class_weight(
            'balanced', np.unique(y_train), y_train)

    train_datagen_args = dict(featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=config['data.rotation_range'],
                              width_shift_range=config['data.width_shift_range'],
                              height_shift_range=config['data.height_shift_range'],
                              horizontal_flip=config['data.horizontal_flip'],
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

    data = dict(train_datagen=train_datagen, train_datagen_args=train_datagen_args, train_size=len(x_train),
                test_datagen=test_datagen, test_datagen_args=test_datagen_args, test_size=len(x_test),
                val_datagen=val_datagen, val_datagen_args=test_datagen_args, val_size=len(x_val),
                nb_classes=nb_classes, image_shape=image_shape, class_weights=class_weights)

    if datagen_flow:
        # create data generators
        data["train_gen"] = train_datagen.flow(x_train, y_train, batch_size=config['data.batch_size'])
        data["test_gen"] = test_datagen.flow(x_test, y_test, batch_size=config['data.batch_size'])
        data["val_gen"] = val_datagen.flow(x_val, y_val, batch_size=config['data.batch_size'])

    if with_datasets:
        data["x_train"], data["y_train"] = x_train, y_train
        data["x_test"], data["y_test"] = x_test, y_test
        data["x_val"], data["y_val"] = x_val, y_val

    return data
