#!/usr/bin/env python

import argparse
from sklearn.model_selection import train_test_split
from src.utils.model_selection import train_test_split_balanced

from .ciarp import extract_ciarp_classes
from .lsa16 import extract_lsa16_classes
from .rwth import extract_rwth_classes

def store_split(x, y, path):
    f = open(path, 'w')
    for img, label in zip(x, y):
        f.write("{} {}".format(img, label))
    f.close()

def generate_splits(args):
    output_dir = args['output']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The data, split between train and test sets:
    if args['dataset'] == "Ciarp":
        x, y = extract_ciarp_classes(args)
    elif args['dataset'] == "lsa16":
        x, y = extract_lsa16_classes(args)
    elif args['dataset'] == "rwth":
        x, y = extract_rwth_classes(args)
    else:
        raise ValueError("Unknow dataset: {}".format(args['dataset']))

    split = train_test_split if n_train_per_class <= 0 else train_test_split_balanced

    if n_train_per_class <= 0:
        x_train, x_test, y_train, y_test = split(x, y, train_size=train_size, test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = split(x, y, n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)

    store_split(x_test, y_test, os.path.join(output_dir, args['split'], 'test.txt'))
    store_split(x_train, y_train, os.path.join(output_dir, args['split'], 'train.txt'))
    store_split(x_train + x_val, y_train + y_val, os.path.join(output_dir, args['split'], 'trainval.txt'))
    store_split(x_val, y_val, os.path.join(output_dir, args['split'], 'val.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate splits for datasets")

    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--output", type=str, default="", help="Output dir to save the splits")
    parser.add_argument("--ext", type=str, default="png", help="File extension")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name")
    parser.add_argument("--train_size", type=float, default=0.75)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--n_train_per_class", type=int, default=0)
    parser.add_argument("--n_test_per_class", type=int, default=0)

    args = vars(parser.parse_args())

    generate_splits(args)
