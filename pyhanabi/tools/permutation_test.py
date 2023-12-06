import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy
from datetime import datetime
import numpy as np
import random
import subprocess


SPLIT_TYPE = {
    "sad": [
        "one", 
        "six", 
        "eleven"
    ],
    "iql": [
        # "one", 
        "six", 
        # "ten"
    ],
    "op": [
        # "one", 
        "six", 
        # "ten"
    ]
}

SPLIT_INDICES = {
    "sad": {
        "one": "0-12", 
        "six": "0-9", 
        "eleven": "0-9"
    },
    "iql": {
        "one": "0-11", 
        "six": "0-9", 
        "ten": "0-9"
    },
    "op": {
        "one": "0-11", 
        "six": "0-9", 
        "ten": "0-9"
    }
}

SPLIT_NAME = {
    "sad": {
        "one": "1-12_Splits", 
        "six": "6-7_Splits", 
        "eleven": "11-2_Splits"
    },
    "iql": {
        "one": "1-11_Splits", 
        "six": "6-6_Splits", 
        "ten": "10-2_Splits"
    },
    "op": {
        "one": "1-11_Splits", 
        "six": "6-6_Splits", 
        "ten": "10-2_Splits"
    }
}

NUM_SPLITS = {
    "sad": {
        "one": 13,
        "six": 10,
        "eleven": 10
    },
    "iql": {
        "one": 12,
        "six": 10,
        "ten": 10
    },
    "op": {
        "one": 12,
        "six": 10,
        "ten": 10
    }
}

METHODS = ["sba", "br"]


def run_permutation_tests(args):
    print()
    generate_code(args)


def generate_code(args):
    jobs = []

    for aht_model in args.aht_models.split(","):
        for split_type in SPLIT_TYPE[aht_model]:
            for partner_model in args.partner_models.split(","):
                if partner_model == aht_model:
                    partner_model += "_aht"
                var_names = []
                var_name_no_method = f"{aht_model}_{split_type}_vs_{partner_model}"
                for method in METHODS:
                    scores = get_scores(args, aht_model, split_type, 
                            partner_model, method)
                    var_name = f"{method}_{var_name_no_method}"
                    print(f"{var_name} = ", scores)
                    var_names.append(var_name)

                print(f"{var_name_no_method} = permutation_test({var_names[0]}, {var_names[1]})")
                print(f"print(f\"{var_name_no_method}: {{{var_name_no_method}:.3f}}\")")
                print()
    return jobs


def get_scores(args, aht_model, split_type, partner_model, method):
    dir_path = os.path.join(
        args.dir,
        aht_model,
        SPLIT_NAME[aht_model][split_type],
        partner_model,
        f"{method}_{aht_model}"
    )

    aht_model_names = get_aht_model_names(aht_model, split_type, method)

    scores = []
    for split_index in range(NUM_SPLITS[aht_model][split_type]):
        partner_model_names = get_model_names(partner_model, split_type, split_index)
        aht_model_name = aht_model_names[split_index]

        pair_scores = []
        for partner_model_name in partner_model_names:
            file_name = f"{aht_model_name}_vs_{partner_model_name}.csv"
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                print("Error: {file_path} doesn't exist.")
                continue

            pair_scores.append(np.mean(np.genfromtxt(file_path, delimiter=',')))
        scores.append(np.mean(pair_scores))

    return scores


def get_aht_model_names(model, split_type, method):
    model_names = []

    splits = load_json_list(f"train_test_splits/{model}" + 
                            f"_splits_{split_type}.json")

    for split_index in range(NUM_SPLITS[model][split_type]):
        indices = [ x + 1 for x in splits[split_index]["train"] ]
        indices_str = '_'.join(str(x) for x in indices)
        model_name = f"{method}_{model}_{split_type}_{indices_str}"
        model_names.append(model_name)

    return model_names

def get_model_names(model, split_type, split_index):
    model_names = []

    if "aht" in model:
        model = model.split("_")[0]
        splits = load_json_list(f"train_test_splits/{model}" + 
                                f"_splits_{split_type}.json")
        model_indices = splits[split_index]["test"]
    else: 
        model_indices = list(range(len(load_json_list(
            f"agent_groups/all_{model}.json"))))

    for model_index in model_indices:
        model_name = f"{model}_{model_index + 1}"
        model_names.append(model_name)

    return model_names


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aht_models", type=str, default="sad,iql,op")
    parser.add_argument("--partner_models", type=str, default="sad,iql,op,obl")
    parser.add_argument("--methods", type=str, default="br,sba")
    parser.add_argument("--dir", type=str, default="aht_scores")
    return  parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_permutation_tests(args)


