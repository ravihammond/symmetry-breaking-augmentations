import os
import subprocess
import argparse
import json
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import pprint
pprint = pprint.pprint


LOAD_DIR = "/app/pyhanabi/wandb_data"
SPLITS_DIR = "/app/pyhanabi/train_test_splits"


def plot_mean_logs(args):
    all_names = get_names(args)

    data = load_data(args, all_names)
    # pprint(data)

    plot_data(args, data)


def get_names(args):
    split_ids = [int(x) for x in args.splits.split(",")]
    splits_path = os.path.join(SPLITS_DIR, args.splits_file + ".json")
    splits = load_json_list(splits_path)
    models = args.models.split(",")

    all_names = {}

    for model in models:

        model_names = []

        for split_i in split_ids:
            indexes = splits[split_i]["train"]
            indexes = [x + 1 for x in indexes]
            idx_str = '_'.join(str(x) for x in indexes)

            name = f"{model}_{idx_str}"
            model_names.append(name)
            all_names[model] = model_names

    return all_names


def load_data(args, all_names):
    all_data = {}

    for model_name in all_names:
        all_model_data = []

        for name in all_names[model_name]:
            load_path = os.path.join(LOAD_DIR, name + ".csv")
            data = genfromtxt(load_path, delimiter=',')
            data = np.delete(data, 0, axis=0)

            all_model_data.append(data)

        mean_data = np.mean(all_model_data, axis=0)
        print(mean_data)
        for i, word in enumerate(["test", "train"]):
            model_name_full = model_name + "_" + word

            mean = mean_data[:, i]
            stderr_high = mean + mean_data[:, i + 2]
            stderr_low = mean - mean_data[:, i + 2]

            all_data[model_name_full + "_mean"] = mean
            all_data[model_name_full + "_stderr_high"] = stderr_high
            all_data[model_name_full + "_stderr_low"] = stderr_low

    return all_data


def plot_data(args, data):
    epochs = np.arange(0, args.num_samples)

    test_idx = 0
    train_idx = 1

    data_type_str = ""
    if args.test:
        data_type_str = "test"
    if args.train:
        data_type_str = "train"

    for dataset in ["train", "test"]:
        if dataset is "train" and not args.train:
            continue
        if dataset is "test" and not args.test:
            continue

        for model in args.models.split(","):
            model_name = model + "_" + dataset
            mean = data[model_name + "_mean"]
            stderr_high = data[model_name + "_stderr_high"]
            stderr_low = data[model_name + "_stderr_low"]
             
            label = [l for l in args.labels.split(",") if l.lower() in model][0]
            label = label + " " + dataset.capitalize()
            plt.plot(epochs, mean, label=label)
            plt.fill_between(epochs, stderr_high, stderr_low, alpha=0.3)

    plt.title(f"SBA vs BR {data_type_str} Scores, 1-12 splits")
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
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--splits_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.parse_args()
    args = parser.parse_args()

    plot_mean_logs(args)

