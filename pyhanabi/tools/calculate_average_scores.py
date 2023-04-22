import os
import sys
import time
import argparse
from collections import defaultdict
import pprint
pprint = pprint.pprint
import numpy as np
import json
import pathlib
import csv

SPLIT_NAME = {
    "six": "6-7_Splits", 
    "one": "1-12_Splits",
    "eleven": "11-2_Splits",
}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def calculate_all_average_scores(args):
    for model in args.model:
        if args.verbose:
            print(f"\n=== Model: {model.upper()} ===")
        for split_type in args.split_type:
            if (len(args.split_type) > 1) and args.verbose:
                print(f"\n===== {SPLIT_NAME[split_type]} =====")
            for data_type in args.data_type:
                if (len(args.data_type) > 1) and args.vervose:
                    print(f"\n--- {data_type.upper()} ---")

                calculate_average_scores(args, model, split_type, data_type)


def calculate_average_scores(args, model, split_type, data_type):
    if args.verbose:
        print()
    dir_path = os.path.join(
        args.dir,
        SPLIT_NAME[split_type],
        data_type,
        f"{model}_scores"
    )

    if not os.path.exists(dir_path):
        if args.verbose:
            print(bcolors.FAIL + f"Path {dir_path} does not exist." + bcolors.ENDC)
        return 

    all_file_names = os.listdir(dir_path)

    splits = load_json_list(f"train_test_splits/sad_splits_{split_type}.json")

    all_means = []

    for split_index in args.split_index:
        if (len(args.split_index) > 1) and args.verbose:
            print(f"\n- SPLIT: {split_index} -")

        split_scores = []

        indexes = splits[split_index]["train"]
        indexes = [x + 1 for x in indexes]
        indexes_str = '_'.join(str(x) for x in indexes)
        model_name = f"{model}_sad_{split_type}_{indexes_str}"

        test_indexes = splits[split_index]["test"]
        for partner_index in range(len(test_indexes)):
            partner_num = test_indexes[partner_index]
            partner_name = f"sad_{partner_num + 1}"
            pair_str = f"{model_name}_vs_{partner_name}_"
            if args.verbose:
                print(partner_name)

            file_name = [ x for x in all_file_names if pair_str in x ]

            if len(file_name) == 0:
                if argse.verbose:
                    print(bcolors.FAIL + f"File {pair_str} does not exist." + bcolors.ENDC)
                continue 
            elif len(file_name) > 1:
                if args.verbose:
                    print(bcolors.WARNING + f"Multiple {pair_str} files exist." + bcolors.ENDC)
                continue 

            file_path = os.path.join(dir_path, file_name[0])

            scores = []

            with open(file_path, newline='') as file:
                reader = csv.reader(file)
                scores = [int(x[0]) for x in list(reader)]

            if len(scores) == 0:
                if argse.verbose: 
                    print(bcolors.FAIL + f"no scores." + bcolors.ENDC)
                continue 

            mean = np.mean(scores)
            sem = np.std(scores) / np.sqrt(len(scores))
            if args.verbose:
                print(f"{mean:.3f} ± {sem:.3f}")

            split_scores = [*split_scores, *scores]

        print(f"\n{model_name}")
        if len(split_scores) == 0:
            print(bcolors.FAIL + f"no scores." + bcolors.ENDC)
            continue 

        mean = np.mean(split_scores)
        all_means.append(mean)
        sem = np.std(split_scores) / np.sqrt(len(split_scores))
        print(f"{mean} ± {sem:.3f}")

    print(f"\n{model} total")
    if len(all_means) == 0:
        print(bcolors.FAIL + f"no scores." + bcolors.ENDC)
        return 

    mean = np.mean(all_means)
    sem = np.std(all_means) / np.sqrt(len(all_means))
    print("len:", len(all_means))
    print(f"{mean:.6f} ± {sem:.6f}")
            

def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="game_data")
    parser.add_argument("--model", type=str, default="br")
    parser.add_argument("--split_index", type=str, default="0")
    parser.add_argument("--split_type", type=str, default="six")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--seeds", type=str, default="0-100")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    args.model = args.model.split(",")
    args.split_index = [ int(x) for x in args.split_index.split(",") ]
    args.split_type = args.split_type.split(",")
    args.data_type = args.data_type.split(",")

    if '-' in args.seeds:
        seed_range = [ int(x) for x in args.seeds.split('-') ]
        assert(len(seed_range) == 2)
        args.seeds = list(np.arange(*seed_range))
    else:
        args.seeds = [ int(x) for x in args.seeds.split(',') ]

    calculate_all_average_scores(args)

