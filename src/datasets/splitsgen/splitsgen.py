#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.model_selection import train_test_split_balanced

from .ciarp import load_ciarp
from .lsa16 import load_lsa16
from .rwth import load_rwth

def store_split(x, y, path, data_dir, mode='w'):
    f = open(path, mode)
    for img, label in zip(x, y):
        img = os.path.relpath(img, data_dir)
        f.write("{} {}\n".format(img, label))
    f.close()

def generate_splits(split, data_dir, splits_dir, dataset, version, train_size, test_size, n_train_per_class, n_test_per_class, seed):
    np.random.seed(seed)

    data_dir = data_dir if data_dir else '/tf/data/{}/data'.format(dataset)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    output_dir = os.path.join(splits_dir, split)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The data, split between train and test sets:
    if dataset == "Ciarp":
        x, y = load_ciarp(data_dir, dataset, version)
    elif dataset == "lsa16":
        x, y = load_lsa16(data_dir, dataset, version)
    elif dataset == "rwth":
        x, y = load_rwth(data_dir, dataset, version)
    else:
        raise ValueError("Unknow dataset: {}".format(dataset))

    split = train_test_split if n_train_per_class <= 0 else train_test_split_balanced

    if n_train_per_class <= 0:
        x_train, x_test, y_train, y_test = split(x, y, train_size=train_size, test_size=test_size, stratify=y, random_state=seed)
    else:
        n_train_per_class = np.floor(n_train_per_class * 1.2)
        x_train, x_test, y_train, y_test = split(np.array(x), np.array(y), train_size=train_size, test_size=test_size,
                                                 n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class, n_dim=False)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train, random_state=seed)

    store_split(x_test, y_test, os.path.join(output_dir, 'test.txt'), data_dir)
    store_split(x_train, y_train, os.path.join(output_dir, 'train.txt'), data_dir)
    store_split(x_val, y_val, os.path.join(output_dir, 'val.txt'), data_dir)    
    store_split(x_train, y_train, os.path.join(output_dir, 'trainval.txt'), data_dir)
    store_split(x_val, y_val, os.path.join(output_dir, 'trainval.txt'), data_dir, 'a+')
