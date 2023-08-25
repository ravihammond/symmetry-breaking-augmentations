import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy
from datetime import datetime

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from tools.eval_model_verbose import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy1", type=str, required=True)
    parser.add_argument("--policy2", type=str, required=True)
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--sad_legacy1", type=int, default=0)
    parser.add_argument("--sad_legacy2", type=int, default=0)
    parser.add_argument("--shuffle_index", type=str, default="0")
    parser.add_argument("--name1", type=str, default="<none>")
    parser.add_argument("--name2", type=str, default="<none>")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    save_scores(args)

