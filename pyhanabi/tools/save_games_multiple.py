import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from save_games import save_games


def save_games_multiple(args):
    run_args = []
    if args.crossplay:
        run_args = generate_jobs_crossplay(args)
    else:
        run_args = generate_jobs(args)
    run_save_games(args, run_args)


def generate_jobs_crossplay(args):
    assert(len(args.model1) == 1)

    run_args = []

    partner_types = ["crossplay_self", "crossplay_other"]
    indexes = list(range(len(load_json_list(
        f"agent_groups/all_{args.model2}.json")))) 

    for partner_type in partner_types:
        run = edict()
        run.data_type = partner_type
        run.sad_legacy = [0,0]
        run.player_name = ["",""]
        model1 = args.model1[0]
        (run.model1, 
         run.sad_legacy[0], 
         run.player_name[0]) = model_to_weight(
                 args, model1, args.crossplay_index)

        for partner_i in indexes:
            if partner_type == "crossplay_self":
                if args.crossplay_index != partner_i:
                    continue
            if partner_type == "crossplay_other":
                if args.crossplay_index == partner_i:
                    continue

            run = copy.copy(run)

            (run.model2, 
             run.sad_legacy[1], 
             run.player_name[1]) = model_to_weight(args, args.model2, partner_i)

            run.out = os.path.join(args.out, model1)
            save_dir = os.path.join(
                args.out, 
                f"{model1}_crossplay",
                partner_type, 
                "games"
            )

            run.out = save_dir
            run_args.append(run)

    return run_args


def generate_jobs(args):
    run_args = []

    split_name = {"six": "6-7_Splits", "one": "1-12_Splits"}
    sad_splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")

    if args.model2 == "sad":
        partner_types = ["test", "train"]
        indexes_map = { x: sad_splits[args.split_index][x] for x in partner_types }

    elif args.model2 == "obl":
        partner_types = ["obl"]
        indexes_map = { "obl": list(range(len(load_json_list("agent_groups/all_obl.json")))) }

    elif args.model2 == "op":
        partner_types = ["op"]
        indexes_map = { "op": list(range(len(load_json_list("agent_groups/all_op.json")))) }

    for partner_type in partner_types:
        run = edict()
        indexes = indexes_map[partner_type]
        run.data_type = partner_type

        for partner_i in indexes:
            run = copy.copy(run)
            run.sad_legacy = [0,0]
            run.player_name = ["",""]
            (run.model2, 
             run.sad_legacy[1], 
             run.player_name[1]) = model_to_weight(args, args.model2, partner_i)

            for model1 in args.model1:
                run = copy.copy(run)
                (run.model1, 
                 run.sad_legacy[0], 
                 run.player_name[0]) = model_to_weight(
                         args, model1, args.crossplay_index)

                run.out = os.path.join(args.out, model1)
                save_dir = os.path.join(args.out, 
                        split_name[args.split_type], 
                        partner_type, 
                        f"{model1}_games"
                )

                run.out = save_dir
                run_args.append(run)


    return run_args


def model_to_weight(args, model, policy_i):
    splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")
    indexes = splits[args.split_index]["train"]
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

    return path, sad_legacy, player_name


def run_save_games(args, run_args):
    for run in run_args:
        input_args = edict({
            "weight1": run.model1,
            "weight2": run.model2,
            "out": run.out,
            "player_name": run.player_name,
            "data_type": run.data_type,
            "sad_legacy": run.sad_legacy,
            "device": args.device,
            "num_game": args.num_game,
            "num_thread": args.num_thread,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "verbose": args.verbose,
            "save": args.save,
        })
        save_games(input_args)
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
    parser.add_argument("--out", type=str, default="game_data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="six")
    parser.add_argument("--crossplay", type=int, default=0)
    parser.add_argument("--crossplay_index", type=int, default=0)
    args = parser.parse_args()

    # batch size is double the number of games
    if args.batch_size is None:
        args.batch_size = args.num_game * 2

    args.model1 = args.model1.split(",")

    return args

if __name__ == "__main__":
    args = parse_args()
    save_games_multiple(args)
