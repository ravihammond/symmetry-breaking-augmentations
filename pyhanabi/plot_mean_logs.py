import os
import sys
import subprocess
import argparse
import json
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import pprint
pprint = pprint.pprint


LOAD_DIR = "/app/pyhanabi/wandb_data"
SPLIT_NAME = {
    "six": "6-7 Splits", 
    "one": "1-12 Splits", 
    "eleven": "11-2 Splits"
}


def plot_mean_logs(args):
    model_names = get_names(args)
    pprint(model_names)

    data = load_data(args, model_names)

    plot_data(args, data)


def get_names(args):
    split_ids = parse_number_list(args.splits)
    splits_file = f"train_test_splits/sad_splits_{args.split_type}.json"
    splits = load_json_list(splits_file)

    model_names = []

    for split_i in split_ids:
        indexes = splits[split_i]["train"]
        indexes = [x + 1 for x in indexes]
        idx_str = '_'.join(str(x) for x in indexes)

        name = f"{args.model}_sad_{args.split_type}_{idx_str}"
        model_names.append(name)

    return model_names

def parse_number_list(game_seeds):
    if '-' in game_seeds:
        seed_range = [int(x) for x in game_seeds.split('-')]
        assert(len(seed_range) == 2)
        game_seed_list = list(np.arange(*seed_range))
    else:
        game_seed_list = [int(x) for x in game_seeds.split(',')]

    return game_seed_list


def load_data(args, model_names):
    all_data = {}

    all_model_data = []

    for name in model_names:
        load_path = os.path.join(LOAD_DIR, name + ".csv")
        data = genfromtxt(load_path, delimiter=',')
        data = np.delete(data, 0, axis=0)

        all_model_data.append(data)

    mean_data = np.mean(all_model_data, axis=0)
    sem_data = np.std(all_model_data, axis=0) / np.sqrt(len(all_model_data))
    print("mean data:", mean_data.shape)
    print(mean_data)
    print("sem data:", sem_data.shape)
    print(sem_data)

    # for i, word in enumerate(["test", "train"]):
    for i, dataset in enumerate(["test", "train"]):
        mean = mean_data[:, i]
        stderr_high = mean + sem_data[:, i]
        stderr_low = mean - sem_data[:, i]

        all_data[f"{dataset}_mean"] = mean
        all_data[dataset + "_stderr_high"] = stderr_high
        all_data[dataset + "_stderr_low"] = stderr_low

    return all_data


def plot_data(args, data):
    epochs = np.arange(0, args.num_samples)

    test_idx = 0
    train_idx = 1

    datasets = []
    if args.test:
        datasets.append("test")
    if args.train:
        datasets.append("train")
    colours = {"train": "blue", "test": "green"}

    for dataset in datasets:
        mean = data[f"{dataset}_mean"]
        stderr_high = data[f"{dataset}_stderr_high"]
        stderr_low = data[f"{dataset}_stderr_low"]
         
        plt.plot(epochs, mean, label=dataset, color=colours[dataset])
        plt.fill_between(epochs, stderr_high, stderr_low, 
                alpha=0.3, color=colours[dataset])

    split_name = SPLIT_NAME[args.split_type]
    plt.title(f"{args.model.upper()} Training Curves, {split_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="br")
    parser.add_argument("--splits", type=str, default="0")
    parser.add_argument("--split_type", type=str, default="six")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--train', type=int, default=1)
    parser.parse_args()
    args = parser.parse_args()

    plot_mean_logs(args)

