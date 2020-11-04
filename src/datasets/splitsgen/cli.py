#!/usr/bin/env python

import argparse
from tf_tools.datasets.splitsgen import generate_splits

from src.datasets.splitsgen.ciarp import load_ciarp
from src.datasets.splitsgen.lsa16 import load_lsa16
from src.datasets.splitsgen.rwth import load_rwth

parser = argparse.ArgumentParser(description="Generate splits for datasets")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--split", type=str)
parser.add_argument("--data_dir", type=str, default="",
                    help="Input dir to load data")
parser.add_argument("--splits_dir", type=str,
                    default="/tmp/splits", help="Output dir to save the splits")
parser.add_argument("--dataset", type=str, default="", help="Dataset name")
parser.add_argument("--version", type=str, default="", help="Dataset version")
parser.add_argument("--train_size", type=float, default=0)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_train_per_class", type=int, default=0)
parser.add_argument("--n_test_per_class", type=int, default=0)
parser.add_argument("--balanced", type=bool, default=False)

args = vars(parser.parse_args())

# The data, split between train and test sets:
if args['dataset'] == "Ciarp":
    x, y = load_ciarp(args['data_dir'], args['dataset'], args['version'])
elif args['dataset'] == "lsa16":
    x, y = load_lsa16(args['data_dir'], args['dataset'], args['version'])
elif args['dataset'] == "rwth":
    x, y = load_rwth(args['data_dir'], args['dataset'], args['version'])
else:
    raise ValueError("Unknow args['dataset']: {}".format(args['dataset']))

generate_splits(x, y, **args)
