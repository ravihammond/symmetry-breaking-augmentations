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
import pandas as pd
import random

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from tools.eval_model_verbose import evaluate_model

NUM_SPLITS = {"six": 10, "one": 13, "eleven": 10}
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
NUM_COLOUR_SHUFFLES = 120
NUM_PARTNERS = {"sad": 13, "op": 12, "obl": 5}

def save_scores(args):
    if args.crossplay:
        if args.model1 == args.model2:
            run_args = generate_jobs_crossplay_self(args)
        else: 
            run_args = generate_jobs_crossplay(args)
    else:
        run_args = generate_jobs(args)

    df = run_save_scores(args, run_args)
    save_data(args, df)


def generate_jobs_crossplay(args):
    assert(len(args.model1) == 1)
    assert(len(args.model2) == 1)

    run_args = []

    model_weights1 = load_json_list(f"agent_groups/all_{args.model1[0]}.json")
    model_weights2 = load_json_list(f"agent_groups/all_{args.model2[0]}.json")

    for index_1 in range(len(model_weights1)):
        run = edict()
        run.sad_legacy = [0,0]
        run.iql_legacy = [0,0]
        model1 = args.model1[0]
        run.player1_index = index_1
        (run.model1, 
         run.sad_legacy[0], 
         run.iql_legacy[0], 
         run.player_name_1) = model_to_weight(
                 args, args.model1[0], index_1, 0)

        for index_2 in range(len(model_weights2)):
            run2 = copy.copy(run)

            run2.player1_index = index_2
            (run2.model2, 
             run2.sad_legacy[1], 
             run2.iql_legacy[1], 
             run2.player_name_2) = model_to_weight(
                     args, args.model2[0], index_2, 0)

            file_name = f"{run2.player_name_1}_vs_{run2.player_name_2}"
            save_dir = os.path.join(
                args.out, 
                f"{args.model1[0]}_vs_{args.model2[0]}",
                "all", 
                "scores"
            )
            file_path = os.path.join(save_dir, file_name)
            run2.csv_name = "None"

            if args.sba:
                for colour_shuffle in range(NUM_COLOUR_SHUFFLES):
                    run3 = copy.copy(run2)
                    run_colour_shuffle = colour_shuffle
                    if args.save:
                        run3.csv_name = file_path
                    run_args.append(run3)
            else:
                if args.save:
                    run2.csv_name = file_path
                run_args.append(run2)

    return run_args


def generate_jobs_crossplay_self(args):
    assert(len(args.model1) == 1)
    assert(len(args.model2) == 1)

    run_args = []

    partner_types = ["crossplay_self", "crossplay_other"]
    if args.sba:
        partner_types = ["crossplay_other"]
    indexes = list(range(len(load_json_list(
        f"agent_groups/all_{args.model2[0]}.json")))) 

    completed_pairs = set()

    for crossplay_index in indexes:
        for partner_type in partner_types:
            run = edict()
            run.sad_legacy = [0,0]
            run.iql_legacy = [0,0]
            run.player1_index = crossplay_index
            model1 = args.model1[0]
            (run.model1, 
             run.sad_legacy[0], 
             run.iql_legacy[0], 
             run.player_name_1) = model_to_weight(
                     args, model1, crossplay_index, 0)

            for partner_i in indexes:
                if partner_type == "crossplay_self":
                    if crossplay_index != partner_i:
                        continue
                if partner_type == "crossplay_other":
                    if crossplay_index == partner_i:
                        continue
                    if args.sba:
                        pair_str = f"{crossplay_index}-{partner_i}"
                        if pair_str in completed_pairs:
                            continue
                        pair_str_reversed = f"{partner_i}-{crossplay_index}"
                        completed_pairs.add(pair_str_reversed)

                run2 = copy.copy(run)
                run2.player2_index = partner_i

                (run2.model2, 
                 run2.sad_legacy[1], 
                 run2.iql_legacy[1], 
                 run2.player_name_2) = model_to_weight(
                         args, args.model2[0], partner_i, 0)

                file_name = f"{run2.player_name_1}_vs_{run2.player_name_2}"
                save_dir = os.path.join(
                    args.out, 
                    f"{model1}_crossplay",
                    partner_type, 
                    "scores"
                )
                file_path = os.path.join(save_dir, file_name)

                run2.csv_name = "None"
                if args.sba:
                    for colour_shuffle in range(NUM_COLOUR_SHUFFLES):
                        run3 = copy.copy(run2)
                        run3.colour_shuffle = colour_shuffle
                        if args.save:
                            run3.csv_name = file_path
                        run_args.append(run3)
                else:
                    if args.save:
                        run2.csv_name = file_path
                    run_args.append(run2)

    return run_args


def generate_jobs(args):
    run_args = []

    split_name = {
        "six": "6-7_Splits", 
        "one": "1-12_Splits",
        "eleven": "11-2_Splits"
    }
    num_splits = {"six": 10, "one": 13, "eleven": 10}
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
                    run.csv_name = "None"
                    if args.save:
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
        iql_legacy = 0

    elif model == "sba":
        player_name = f"sba_sad_{args.split_type}_{indexes_str}"
        path = f"exps/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0
        iql_legacy = 0

    elif "obl" in model:
        policies = load_json_list("agent_groups/all_obl.json")
        path = policies[policy_i]
        player_name = f"obl_{policy_i + 1}"
        sad_legacy = 0
        iql_legacy = 0

    elif model == "op":
        policies = load_json_list("agent_groups/all_op.json")
        path = policies[policy_i]
        player_name = f"op_{policy_i + 1}"
        sad_legacy = 1
        iql_legacy = 0

    elif model == "sad":
        policies = load_json_list("agent_groups/all_sad.json")
        path = policies[policy_i]
        player_name = f"sad_{policy_i + 1}"
        sad_legacy = 1
        iql_legacy = 0

    elif model == "iql":
        policies = load_json_list("agent_groups/all_iql.json")
        path = policies[policy_i]
        player_name = f"iql_{policy_i + 1}"
        sad_legacy = 1
        iql_legacy = 1

    assert(os.path.exists(path))

    return path, sad_legacy, iql_legacy, player_name


def print_mean_sem(data, name):
    mean = np.mean(data)
    sem = np.std(data) / np.sqrt(len(data))
    print(f"{name}: {mean:.2f} ± {sem:.2f}")

def run_save_scores(args, run_args):
    df  = pd.DataFrame()
    sp_scores = []
    sp_bombout = []
    xp_scores = []
    xp_bombout = []

    for run in run_args:
        df_row, scores, bombout = run_job(args, run)
        df = pd.concat([df, df_row], ignore_index=True)
        if run.model1 == run.model2 and \
                run.player1_index == run.player2_index:
            sp_scores.append(scores)
            sp_bombout.append(bombout)
        else:
            xp_scores.append(scores)
            xp_bombout.append(bombout)

    if not args.sba:
        print()
        print(f"{args.model1[0]} vs {args.model2[0]}")
        print()

        if len(sp_scores) > 0:
            print("selfplay")
            print_mean_sem(sp_scores, "scores")
            print_mean_sem(sp_bombout, "bombout")
            print()

        print("crossplay")
        print_mean_sem(xp_scores, "scores")
        print_mean_sem(xp_bombout, "bombout")
        print()

        print_mean_sem(sp_scores + xp_scores, "scores")
        print_mean_sem(sp_bombout + xp_bombout, "bombout")
        print()

    print(df.to_string())
    print(df.info())

    return df


def run_job(args, run):
    df = pd.DataFrame()

    model1_shuffle_index = f"({run.colour_shuffle}) " if args.sba else ""
    print(f"\n{run.player_name_1} {model1_shuffle_index}vs {run.player_name_2}")
    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y_%H:%M:%S")
    run.csv_name = f"{run.csv_name}_{date_time}.csv"
    colour_shuffle = run.colour_shuffle if "colour_shuffle" in run else -1

    input_args = edict({
        "weight1": run.model1,
        "weight2": run.model2,
        "csv_name": "None",
        "sad_legacy": run.sad_legacy,
        "iql_legacy": run.iql_legacy,
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
        "shuffle_index1": colour_shuffle,
        "shuffle_index2": -1,
    })

    score, bombout = evaluate_model(input_args)

    df = pd.DataFrame(
        data=[[
            f"{args.model1[0]}_{run.player1_index + 1}",
            f"{args.model2[0]}_{run.player2_index + 1}",
            int(run.colour_shuffle),
            float(score),
        ]],
        columns=[
            "actor1",
            "actor2",
            "shuffle_index",
            "score",
        ]
    )

    return df, score, bombout


def save_data(args, df):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    filename = f"{args.model1[0]}_vs_{args.model2[0]}_scores.pkl"
    filepath = os.path.join(args.out, filename)

    print("Saving:", filepath)
    df.to_pickle(filepath, compression="gzip")


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
    parser.add_argument("--out", type=str, default="game_data_new")
    parser.add_argument("--crossplay", type=int, default=0)
    parser.add_argument("--sba", type=int, default=0)
    parser.add_argument("--save", type=int, default=0)
    args = parser.parse_args()

    args.model1 = args.model1.split(",")
    args.model2 = args.model2.split(",")

    return args

if __name__ == "__main__":
    args = parse_args()
    save_scores(args)

