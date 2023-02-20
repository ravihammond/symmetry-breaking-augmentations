import os
import subprocess
import argparse
import pandas as pd
import wandb
import json
import pprint
pprint = pprint.pprint


PROJECT = "ravihammond/hanabi-conventions"
SAVE_DIR = "/app/pyhanabi/wandb_data"
SPLITS_DIR = "/app/pyhanabi/train_test_splits"
LOG_NAMES = [
    "test_score",
    "train_score",
    "test_score_stderr",
    "train_score_stderr"
]

def export_wandb(args):
    subprocess.call(["wandb", "login", os.environ["WANDB_TOKEN"]])
    api = wandb.Api()

    names = get_names(args)

    pprint(names)

    runs = api.runs(PROJECT)
    for run in runs:
        if run.name not in names:
            continue
        pprint(run.name)
        df = run.history(samples=100000)[LOG_NAMES]
        df = df.head(args.num_samples)
        save_path = os.path.join(SAVE_DIR, run.name + ".csv")
        df.to_csv(save_path, index=False)


def get_names(args):
    split_ids = [int(x) for x in args.splits.split(",")]
    splits_path = os.path.join(SPLITS_DIR, args.splits_file + ".json")
    splits = load_json_list(splits_path)
    models = args.models.split(",")

    names = []

    for split_i in split_ids:
        indexes = splits[split_i]["train"]
        indexes = [x + 1 for x in indexes]
        idx_str = '_'.join(str(x) for x in indexes)

        for model in models:
            name = f"{model}_{idx_str}"
            names.append(name)

    names.sort()

    return names


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--splits_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.parse_args()
    args = parser.parse_args()

    export_wandb(args)

