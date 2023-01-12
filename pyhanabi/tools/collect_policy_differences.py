import os
import sys
import argparse
import json
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from policy_differences import calculate_policy_differences


def collect_policy_differences(args):
    # Get all data collection combinations
    run_args = generate_run_args(args)

    # Run all data collection combinations
    collect_data(args, run_args)


def generate_run_args(args):
    run_args = []

    rollout_policies = load_json_list(args.rollout_policies)
    splits = load_json_list(args.train_test_splits)

    for split_i in args.split_indexes:
        test_indexes = splits[split_i]["test"]
        for i, policy_i in enumerate(test_indexes):
            run = edict()

            # rollout_policy
            run.rollout_policy = rollout_policies[policy_i]

            # rollout_sad_legacy
            run.rollout_sad_legacy = 0
            if any(x in run.rollout_policy for x in ["sad","op"]):
                run.rollout_sad_legacy = 1

            # comp_policy, comp_sad_legacy
            run.comp_policies, run.comp_sad_legacy = get_comp_policies(
                    args, splits, split_i)

            # comp_names
            run.comp_names = args.compare_models

            if args.single_policy is None or i == args.single_policy:
                run_args.append(run)

    return run_args


def get_comp_policies(args, splits, split_i):
    comp_policies = []
    comp_sad_legacy = []
    train_indexes = splits[split_i]["train"]
    train_indexes = [x + 1 for x in train_indexes]
    train_indexes_str = '_'.join(str(x) for x in train_indexes)

    for comp_model in args.compare_models:
        if comp_model == "br":
            path = f"exps/br_sad_{train_indexes_str}/model_epoch1000.pthw"
            sad_legacy = 0

        elif comp_model == "sba":
            path = f"exps/sba_sad_{train_indexes_str}/model_epoch1000.pthw"
            sad_legacy = 0

        elif comp_model == "obl":
            path = "../training_models/obl1/model0.pthw" 
            sad_legacy = 0

        elif comp_model == "op":
            path = "../training_models/op_models/op_1.pthw" 
            sad_legacy = 1

        comp_policies.append(path)
        comp_sad_legacy.append(sad_legacy)

    return comp_policies, comp_sad_legacy


def collect_data(args, run_args):
    for run in run_args:
        calculate_policy_differences(
            run.rollout_policy,
            run.comp_policies,
            args.num_game,
            args.num_thread,
            args.seed,
            args.batch_size,
            run.rollout_sad_legacy,
            run.comp_sad_legacy,
            args.device,
            run.comp_names,
            args.output_dir,
            args.rand_policy,
            args.verbose,
            args.compare_as_base,
        )


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="similarity_data")
    parser.add_argument("--rollout_policies", type=str, required=True)
    parser.add_argument("--train_test_splits", type=str, default="None")
    parser.add_argument("--split_indexes", type=str, default="None")
    parser.add_argument("--single_policy", type=int, default=None)
    parser.add_argument("--compare_models", type=str, default="None")
    parser.add_argument("--compare_as_base", type=str, default="None")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--rand_policy", type=int, default=0)
    args = parser.parse_args()

    if args.compare_models == "None":
        args.compare_models = []
    else:
        args.compare_models = args.compare_models.split(",")

    if args.split_indexes == "None":
        args.split_indexes = []
    else:
        args.split_indexes = [int(x) for x in args.split_indexes.split(",")]

    if args.batch_size is None:
        args.batch_size = args.num_game * 2

    return args

if __name__ == "__main__":
    args = parse_args()
    collect_policy_differences(args)
