import argparse
from src.datasets import generate_splits

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
