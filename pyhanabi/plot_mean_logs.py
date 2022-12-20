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
SPLITS_PATH = "/app/pyhanabi/sad_train_test_splits.json"


def plot_mean_logs(args):
    all_names = get_names(args)

    data = load_data(args, all_names)
    pprint(data)

    plot_data(args, data)


def get_names(args):
    split_ids = [int(x) for x in args.splits.split(",")]
    splits = load_json_list(SPLITS_PATH)
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
    mean_models = {}

    for model_name in all_names:
        all_model_data = []

        for name in all_names[model_name]:
            load_path = os.path.join(LOAD_DIR, name + ".csv")
            data = genfromtxt(load_path, delimiter=',')
            data = np.delete(data, 0, axis=0)

            all_model_data.append(data)

        mean_data = np.mean(all_model_data, axis=0)
        mean_models[model_name] = mean_data

    return mean_models


def plot_data(args, data):
    epochs = np.arange(0, args.num_samples)

    test_idx = 0
    train_idx = 1

    data_idx = -1

    data_type_str = ""
    if args.test:
        data_idx = 0
        data_type_str = "Test"
    if args.train:
        data_idx = 1
        data_type_str = "Train"

    sba_data = data["sba_sad"][:, data_idx]
    br_data = data["br_sad"][:, data_idx]

    plt.title(f"SBA vs BR {data_type_str} Scores")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.plot(epochs, sba_data, label="SBA", color="green")
    plt.plot(epochs, br_data, label="BR", color="blue")
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
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument('--test', action="store_true", )
    parser.add_argument('--train', action="store_true", )
    parser.parse_args()
    args = parser.parse_args()

    print("test:", args.test)
    print("train:", args.train)

    plot_mean_logs(args)

