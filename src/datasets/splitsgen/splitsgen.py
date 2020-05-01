#!/usr/bin/env python

import os
from sklearn.model_selection import train_test_split
from src.utils.model_selection import train_test_split_balanced

from .ciarp import load_ciarp
from .lsa16 import load_lsa16
from .rwth import load_rwth

def store_split(x, y, path):
    f = open(path, 'w')
    for img, label in zip(x, y):
        f.write("{} {}".format(img, label))
    f.close()

def generate_splits(args):
    output_dir = os.path.join(args['output'], args['split'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The data, split between train and test sets:
    if args['dataset'] == "Ciarp":
        x, y = load_ciarp(args)
    elif args['dataset'] == "lsa16":
        x, y = load_lsa16(args)
    elif args['dataset'] == "rwth":
        x, y = load_rwth(args)
    else:
        raise ValueError("Unknow dataset: {}".format(args['dataset']))

    train_size=args['train_size']
    test_size=args['test_size']
    n_train_per_class=args['n_train_per_class']
    n_test_per_class=args['n_test_per_class']

    split = train_test_split if n_train_per_class <= 0 else train_test_split_balanced

    if n_train_per_class <= 0:
        x_train, x_test, y_train, y_test = split(x, y, train_size=train_size, test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = split(x, y, n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)

    store_split(x_test, y_test, os.path.join(output_dir, 'test.txt'))
    store_split(x_train, y_train, os.path.join(output_dir, 'train.txt'))
    store_split(x_train + x_val, y_train + y_val, os.path.join(output_dir, 'trainval.txt'))
    store_split(x_val, y_val, os.path.join(output_dir, 'val.txt'))
