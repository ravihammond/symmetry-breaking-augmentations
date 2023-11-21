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

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from tools.eval_model_verbose import evaluate_model

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


def eval_aht(args):
    alter_args(args)
    jobs = create_jobs(args)


def alter_args(args):
    if "-" in args.split_indices:
        range_nums = [int(x) for x in args.split_indices.split("-")]
        args.split_indices = np.arange(
                range_nums[0], range_nums[1] + 1).tolist()
    else:
        args.split_indices = [int(x) for x in args.split_indices.split(",")]


def create_jobs(args):
    run_args = []

    splits = load_json_list(f"train_test_splits/{args.train_partner_model}_" + 
            f"splits_{args.split_type}.json")

    run = edict()
    run.sad_legacy = [0, 0]
    run.iql_legacy = [0, 0]
    run.player_name = ["", ""]

    for split_index in args.split_indices:
        run1 = copy.copy(run)
        indices = splits[split_index]
        (run1.model1, 
         run1.sad_legacy[0], 
         run1.iql_legacy[0], 
         run1.player_name[0]) = model_to_weight(args, True, 
                 args.aht_model, split_index=split_index)

        partner_indices = []
        if "aht" in args.partner_model:
            partner_indices = indices["test"]
        else:
            partner_indices = list(range(len(load_json_list(
                f"agent_groups/all_{args.partner_model}.json"))))

        for partner_idx in partner_indices:
            run2 = copy.copy(run1)
            (run2.model2, 
             run2.sad_legacy[1], 
             run2.iql_legacy[1], 
             run2.player_name[1]) = model_to_weight(args, False, 
                     args.partner_model, policy_i=partner_idx)

            run2.pair_name = \
                f"{run2.player_name[0]}_vs_{run2.player_name[1]}"
            run2.csv_name = "None"
            if args.save:
                run2.csv_name = os.path.join(
                    args.out, 
                    args.train_partner_model,
                    SPLIT_NAME[args.train_partner_model][args.split_type], 
                    args.partner_model,
                    args.aht_model,
                    f"{run2.pair_name}.csv"
                )

            run_args.append(run2)

    return run_args



def model_to_weight(args, aht_policy, model, split_index=0, policy_i=None):
    splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")
    indices = splits[split_index]["train"]
    indices = [x + 1 for x in indices]
    indices_str = '_'.join(str(x) for x in indices)

    player_name = model

    if aht_policy:
        player_name = f"{model}_{args.split_type}_{indices_str}"
        path = f"exps/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0
        iql_legacy = 0
    else:
        if "aht" in model:
            model = model.split("_")[0]
        policies = load_json_list(f"agent_groups/all_{model}.json")
        path = policies[policy_i]
        player_name = f"{model}_{policy_i + 1}"
        sad_legacy = 0
        iql_legacy = 0
        if model in ["sad", "op", "iql"]:
            sad_legacy = 1
            if model == "iql":
                iql_legacy = 1

    return path, sad_legacy, iql_legacy, player_name


def parse_numbers(numbers):
    if '-' in numbers:
        range = [int(x) for x in numbers.split('-')]
        assert(len(range) == 2)
        numbers_list = list(np.arange(*range))
    else:
        numbers_list = [int(x) for x in numbers.split(',')]

    return numbers_list


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aht_model", type=str, required=True)
    parser.add_argument("--train_partner_model", type=str, required=True)
    parser.add_argument("--partner_model", type=str, required=True)
    parser.add_argument("--split_type", type=str, required=True)
    parser.add_argument("--split_indices", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    eval_aht(args)

