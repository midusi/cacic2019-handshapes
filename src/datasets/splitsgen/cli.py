#!/usr/bin/env python

import argparse
from src.datasets import generate_splits

parser = argparse.ArgumentParser(description="Generate splits for datasets")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--split", type=str)
parser.add_argument("--data_dir", type=str, default="", help="Input dir to load data")
parser.add_argument("--splits_dir", type=str, default="/tmp/splits", help="Output dir to save the splits")
parser.add_argument("--dataset", type=str, default="", help="Dataset name")
parser.add_argument("--version", type=str, default="", help="Dataset version")
parser.add_argument("--train_size", type=float, default=0)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_train_per_class", type=int, default=0)
parser.add_argument("--n_test_per_class", type=int, default=0)
parser.add_argument("--balanced", type=bool, default=False)

args = vars(parser.parse_args())

generate_splits(**args)
