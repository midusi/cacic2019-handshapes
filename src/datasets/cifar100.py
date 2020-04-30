"""Load cifar100 dataset"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.datasets import cifar100

def load_cifar100(dataset_name):
    """
    Load cifar100 dataset.

    Returns (x, y): as dataset x and y.

    """

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)

    return (x_train, y_train), (x_test, y_test)
