import argparse
import os
import sys
import pprint
pprint = pprint.pprint
import re

from easydict import EasyDict

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

from eval_model_verbose import evaluate_model


def run_evaluations(args):
    weights = extract_weight_names(args.folder)
    eval_args = create_eval_model_args(args)

    for weight_file in weights:
        eval_args.weight1 = weight_file
        evaluate_model(eval_args)


def extract_weight_names(folder_path):
    assert os.path.isdir(folder_path)

    weights = []
    for file_name in os.listdir(folder_path):
        if re.match("model_epoch([0-9]+).pthw", file_name):
            weights.append(os.path.join(folder_path, file_name))

    assert len(weights) > 0

    weights.sort(key=natural_keys)
    return weights


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def atoi(text):
    return int(text) if text.isdigit() else text


def create_eval_model_args(args):
    eval_args = EasyDict()
    eval_args.weight2 = args.partner
    eval_args.weight3 = None
    eval_args.output = os.path.join(args.folder, "verbose.log")
    eval_args.num_player = 2
    eval_args.seed = 1
    eval_args.bomb = 0
    eval_args.num_game = args.num_game
    eval_args.num_run = 1
    eval_args.convention = args.convention
    eval_args.convention_sender = args.convention_sender
    eval_args.override1 = args.override1
    eval_args.override2 = args.override2
    return eval_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--partner", type=str, required=True)
    parser.add_argument("--num_game", default=1000, type=int)
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--convention_sender", default=0, type=int)
    parser.add_argument("--override1", default=0, type=int)
    parser.add_argument("--override2", default=0, type=int)
    args = parser.parse_args()
    run_evaluations(args)

