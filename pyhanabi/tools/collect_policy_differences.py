import os
import sys
import argparse
import json
import copy
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from policy_differences import run_policy_evaluation


def collect_policy_differences(args):
    # Get all data collection combinations
    if args.sad_crossplay:
        run_args = generate_crossplay_run_args(args)
    else:
        run_args = generate_split_run_args(args)

    # pprint(run_args)

    # Run all data collection combinations
    collect_data(args, run_args)


def generate_crossplay_run_args(args):
    run_args = []
    rollout_policies = load_json_list(args.rollout_policies)

    run = edict()

    run.rollout_policy1 = rollout_policies[args.single_policy]
    run.base_name = f"sad_{args.single_policy + 1}"

    for i in range(args.index_start, args.index_end):
        if args.single_policy is not None and i == args.single_policy:
            continue

        run = copy.deepcopy(run)

        run.rollout_policy2 = rollout_policies[i]
        run.rollout_sad_legacy = [1, 1]

        run.comp_policies = [rollout_policies[i]]
        run.comp_sad_legacy = [1]

        run_args.append(run)
    
    return run_args

def generate_split_run_args(args):
    run_args = []

    rollout_policies = load_json_list(args.rollout_policies)
    splits = load_json_list(args.train_test_splits)

    for split_i in args.split_indexes:
        test_indexes = splits[split_i][args.split_type]
        for i, policy_i in enumerate(test_indexes):
            if args.single_policy is not None and i != args.single_policy:
                continue

            run = edict()

            run.rollout_policy1 = rollout_policies[policy_i]
            run.rollout_policy2 = rollout_policies[policy_i]
            run.rollout_sad_legacy = [1, 1]

            run.comp_policies = []
            run.comp_sad_legacy = []

            for comp_model in args.compare_models:
                policy, sad_legacy = name_to_policy(
                        comp_model, splits, split_i, 
                        args.name_ext, rollout_policies[policy_i])
                run.comp_policies.append(policy)
                run.comp_sad_legacy.append(sad_legacy)


            for base_model in args.base_models:
                run = copy.deepcopy(run)
                run.rollout_policy1, run.rollout_sad_legacy[0] = name_to_policy(
                        base_model, splits, split_i, args.name_ext, args.name_ext)
                run.act_name = base_model
                run.base_name = base_model
                run_args.append(run)


    return run_args


def name_to_policy(model_name, splits, split_i, name_ext, sad_rollout_policy):
    train_indexes = splits[split_i]["train"]
    train_indexes = [x + 1 for x in train_indexes]
    train_indexes_str = '_'.join(str(x) for x in train_indexes)

    if name_ext != "":
        name_ext += "_"

    if model_name == "br":
        path = f"exps/br_sad_{name_ext}{train_indexes_str}/model_epoch1000.pthw"
        sad_legacy = 0

    elif model_name == "sba":
        path = f"exps/sba_sad_{name_ext}{train_indexes_str}/model_epoch1000.pthw"
        sad_legacy = 0

    elif "obl" in model_name:
        path = f"../training_models/obl1/{model_name}/model0.pthw" 
        sad_legacy = 0

    elif model_name == "op":
        path = "../training_models/op_models/op_1.pthw" 
        sad_legacy = 1

    elif model_name == "sad":
        path = sad_rollout_policy
        sad_legacy = 1

    return path, sad_legacy


def collect_data(args, run_args):
    for run in run_args:
        input_args = edict({
            "act_policy1": run.rollout_policy1,
            "act_policy2": run.rollout_policy2,
            "comp_policies": run.comp_policies,
            "num_game": args.num_game,
            "num_thread": args.num_thread,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "act_sad_legacy": run.rollout_sad_legacy,
            "comp_sad_legacy": run.comp_sad_legacy,
            "device": args.device,
            "comp_names": args.compare_models,
            "outdir": args.output_dir,
            "rand_policy": args.rand_policy,
            "verbose": args.verbose,
            "compare_as_base": args.compare_as_base,
            "base_name": run.base_name,
            "similarity_across_all": args.similarity_across_all,
        })
        run_policy_evaluation(input_args)
        print()


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="similarity_data")
    parser.add_argument("--rollout_policies", type=str, required=True)
    parser.add_argument("--act_comp_name", type=str, default=None)
    parser.add_argument("--train_test_splits", type=str, default="None")
    parser.add_argument("--split_indexes", type=str, default="None")
    parser.add_argument("--split_type", type=str, default="test")
    parser.add_argument("--single_policy", type=int, default=None)
    parser.add_argument("--compare_models", type=str, default="None")
    parser.add_argument("--base_models", type=str, default="None")
    parser.add_argument("--compare_as_base", type=str, default="None")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--rand_policy", type=int, default=0)
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--sad_crossplay", type=int, default=0)
    parser.add_argument("--index_start", type=int, default=-1)
    parser.add_argument("--index_end", type=int, default=-1)
    parser.add_argument("--similarity_across_all", type=int, default=0)
    args = parser.parse_args()

    if args.compare_models == "None":
        args.compare_models = []
    else:
        args.compare_models = args.compare_models.split(",")

    if args.base_models == "None":
        args.base_models = []
    else:
        args.base_models = args.base_models.split(",")

    if args.split_indexes == "None":
        args.split_indexes = []
    else:
        args.split_indexes = [int(x) for x in args.split_indexes.split(",")]

    if args.batch_size is None:
        args.batch_size = args.num_game * 2

    if args.output_dir is not None:
        args.output_dir = os.path.join("similarity_data", args.output_dir)

    return args

if __name__ == "__main__":
    args = parse_args()
    collect_policy_differences(args)
