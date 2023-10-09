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

SPLITS = {
    "one": "0,1,2,3,4,5,6,7,8,9,10,11,12",
    "six": "0,1,2,3,4,5,6,7,8,9",
    "eleven": "0,1,2,3,4,5,6,7,8,9"
}


def plot_all_mean_logs(args):
    args.models = args.models.split(",")
    args.split_types = args.split_types.split(",")

    model_names = get_names(args)

    data = load_data(args, model_names)

    plot_data(args, data)


def get_names(args):
    model_names = {}

    for split_type in args.split_types:
        model_names[split_type] = {}
        for model in args.models:
            model_names[split_type][model] = []
            split_ids = parse_number_list(SPLITS[split_type])
            splits_file = f"train_test_splits/sad_splits_{split_type}.json"
            splits = load_json_list(splits_file)

            for split_i in split_ids:
                indexes = splits[split_i]["train"]
                indexes = [x + 1 for x in indexes]
                idx_str = '_'.join(str(x) for x in indexes)

                name = f"{model}_sad_{split_type}_{idx_str}"
                model_names[split_type][model].append(name)

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
    data = {}

    for split_type in args.split_types:
        data[split_type] = {}
        for model in args.models:
            data[split_type][model] = {}
            names = model_names[split_type][model]
            for data_type in ["train", "test"]:
                scores = np.zeros((args.num_steps, len(names), 1))
                for i, name in enumerate(names):
                    scores_model = load_scores(args, split_type, 
                            model, data_type, name)
                    scores[:, i] = scores_model
                data[split_type][model][data_type] = scores
                    
    return data


def load_scores(args, split_type, model, data_type, name):
    load_path = os.path.join(LOAD_DIR, name + ".csv")
    data = genfromtxt(load_path, delimiter=',')
    data = np.delete(data, 0, axis=0)
    data_type_i = {"train": 1, "test": 0}
    data = data[:, data_type_i[data_type]]
    data = np.expand_dims(data, axis=1)
    return data


def plot_data(args, data):
    if len(data) == 2:
        fig_height = 7.5
    elif len(data) == 3:
        fig_height = 11.3
    fig = plt.figure(constrained_layout=True, figsize=(6.65, fig_height))
    subfigs = fig.subfigures(len(data), 1)

    split_titles = ["a)", "b)", "c)"]
    plot_titles = ["Training", "Testing (AHT)"]
    data_types = ["train", "test"]

    for i, split_type in enumerate(data):
        axes = subfigs[i].subplots(1, len(data[split_type]), sharey=True)

        for j, data_type in enumerate(data_types):
            num_splits = len(parse_number_list(SPLITS[split_type]))
            create_plot(
                args,
                axes[j], 
                data[split_type],
                data_type,
                num_splits,
                plot_titles[j],
                i == 0 and j == 0,
                j == 0,
            )
        subfigs[i].suptitle(split_titles[i], fontsize=16)

    plt.show()


def create_plot(args, ax, data, data_type, num_splits, 
        title, show_legend, show_ylabel):
    mode_labels = ["BR", "SBA"]
    colors = ["#d62728", "#1f77b4"]

    x_vals = np.arange(args.num_steps)
    for i, model in enumerate(data):
        y_vals = data[model][data_type]
        y_mean = y_vals.mean(1).squeeze()
        y_std = y_vals.std(1).squeeze() / np.sqrt(num_splits)
        ax.plot(x_vals, y_mean, colors[i], label=mode_labels[i])
        ax.fill_between(x_vals, y_mean + y_std, y_mean - y_std,
                        color=colors[i], alpha=0.3)
    ax.set_ylim([0, 25])
    ax.set_xlim([0, args.num_steps])

    if show_legend:
        ax.legend(loc="lower right")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    if show_ylabel:
        ax.set_ylabel("Score")


def plot_datax(args, data):
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
        # print("mean")
        # print(mean)
        # print("stderr high")
        # print(stderr_high)
         
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
    parser.add_argument("--models", type=str, default="br,sba")
    parser.add_argument("--split_types", type=str, default="one,six,eleven")
    parser.add_argument("--num_steps", type=int, default=1000)
    # parser.add_argument('--test', type=int, default=1)
    # parser.add_argument('--train', type=int, default=1)
    parser.parse_args()
    args = parser.parse_args()

    plot_all_mean_logs(args)

