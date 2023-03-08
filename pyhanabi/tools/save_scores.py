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

NUM_SPLITS = {"six": 10, "one": 13}
SPLIT_NAME = {"six": "6-7_Splits", "one": "1-12_Splits"}

def save_scores(args):
    if args.crossplay:
        run_args = generate_jobs_crossplay(args)
    else:
        run_args = generate_jobs(args)

    run_save_scores(args, run_args)


def generate_jobs_crossplay(args):
    assert(len(args.model1) == 1)

    run_args = []

    partner_types = ["crossplay_self", "crossplay_other"]
    indexes = list(range(len(load_json_list(
        f"agent_groups/all_{args.model2}.json")))) 

    for crossplay_index in indexes:
        for partner_type in partner_types:
            run = edict()
            run.sad_legacy = [0,0]
            model1 = args.model1[0]
            (run.model1, 
             run.sad_legacy[0], 
             player_name_1) = model_to_weight(
                     args, model1, crossplay_index, 0)

            for partner_i in indexes:
                if partner_type == "crossplay_self":
                    if crossplay_index != partner_i:
                        continue
                if partner_type == "crossplay_other":
                    if crossplay_index == partner_i:
                        continue

                run = copy.copy(run)

                (run.model2, 
                 run.sad_legacy[1], 
                 player_name_2) = model_to_weight(
                         args, args.model2, partner_i, 0)

                file_name = f"{player_name_1}_vs_{player_name_2}"
                save_dir = os.path.join(
                    args.out, 
                    f"{model1}_crossplay",
                    partner_type, 
                    "scores"
                )
                file_path = os.path.join(save_dir, file_name)

                run.csv_name = file_path
                run_args.append(run)

    return run_args


def generate_jobs(args):
    run_args = []

    split_name = {"six": "6-7_Splits", "one": "1-12_Splits"}
    num_splits = {"six": 10, "one": 13}
    sad_splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")

    for split_index in range(num_splits[args.split_type]):
        if args.model2 == "sad":
            partner_types = ["test", "train"]
            indexes_map = { x: sad_splits[split_index][x] for x in partner_types }

        elif args.model2 == "obl":
            partner_types = ["obl"]
            indexes_map = { "obl": list(range(len(load_json_list("agent_groups/all_obl.json")))) }

        elif args.model2 == "op":
            partner_types = ["op"]
            indexes_map = { "op": list(range(len(load_json_list("agent_groups/all_op.json")))) }

        for partner_type in partner_types:
            run = edict()
            indexes = indexes_map[partner_type]

            for sad_i in indexes:
                run = copy.copy(run)

                run.sad_legacy = [0,0]
                (run.model2, 
                 run.sad_legacy[1], 
                 player_name_2) = model_to_weight(args, args.model2, 
                         sad_i, split_index)

                for model1 in args.model1:
                    run = copy.copy(run)
                    (run.model1, 
                     run.sad_legacy[0], 
                     player_name_1) = model_to_weight(args, model1, 
                             sad_i, split_index)
                    file_name = f"{player_name_1}_vs_{player_name_2}"
                    save_dir = os.path.join(args.out, 
                            SPLIT_NAME[args.split_type], 
                            partner_type, f"{model1}_scores")
                    # assert(os.path.exists(save_dir))
                    file_path = os.path.join(save_dir, file_name)
                    run.csv_name = file_path
                    run_args.append(run)

    return run_args


def model_to_weight(args, model, policy_i, split_index):
    splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")
    indexes = splits[split_index]["train"]
    indexes = [x + 1 for x in indexes]
    indexes_str = '_'.join(str(x) for x in indexes)

    player_name = model

    if model == "br":
        player_name = f"br_sad_{args.split_type}_{indexes_str}"
        path = f"exps/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0

    elif model == "sba":
        player_name = f"sba_sad_{args.split_type}_{indexes_str}"
        path = f"exps/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0

    elif "obl" in model:
        policies = load_json_list("agent_groups/all_obl.json")
        path = policies[policy_i]
        player_name = f"obl_{policy_i + 1}"
        sad_legacy = 0

    elif model == "op":
        policies = load_json_list("agent_groups/all_op.json")
        path = policies[policy_i]
        player_name = f"op_{policy_i + 1}"
        sad_legacy = 1

    elif model == "sad":
        policies = load_json_list("agent_groups/all_sad.json")
        path = policies[policy_i]
        player_name = f"sad_{policy_i + 1}"
        sad_legacy = 1

    assert(os.path.exists(path))

    return path, sad_legacy, player_name


def run_save_scores(args, run_args):
    for run in run_args:
        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y_%H:%M:%S")
        run.csv_name = f"{run.csv_name}_{date_time}.csv"
        input_args = edict({
            "weight1": run.model1,
            "weight2": run.model2,
            "csv_name": run.csv_name,
            "sad_legacy": run.sad_legacy,
            "num_game": args.num_game,
            # other needed
            "weight3": None,
            "output": None,
            "num_player": 2,
            "seed": 0,
            "bomb": 0,
            "num_run": 1,
            "device": "cuda:0",
            "convention": "None",
            "convention_index": None,
            "override0": 0,
            "override1": 0,
            "belief_stats": 0,
            "belief_model": "None",
            "partner_models": "None",
            "train_test_splits": None,
            "split_index": 0,
        })
        evaluate_model(input_args)
        print()


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--split_type", type=str, default="six")
    parser.add_argument("--out", type=str, default="game_data")
    parser.add_argument("--crossplay", type=int, default=0)
    args = parser.parse_args()

    args.model1 = args.model1.split(",")

    return args

if __name__ == "__main__":
    args = parse_args()
    save_scores(args)
