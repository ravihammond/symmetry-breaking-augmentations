import os
import sys
import argparse
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import json
from os import listdir
from os.path import isfile, join
from pprint import pprint
# from statsmodels.stats.proportion import proportion_confint

SAMPLES_PER_PAIR = 5000


def plot_differences(args):
    all_data = generate_plot_data(args)
    plot_data(args,all_data)


def generate_plot_data(args):
    all_data = []
    for comp_model in args.compare_models:
        data = edict()
        data.name = comp_model
        
        if comp_model in ["br", "sba"]:
            data_files = get_data_files_from_splits(args, comp_model)
        elif comp_model == "sad":
            data_files = get_all_sad_data_files(args)
        else:
            data_files = get_all_data_files(args, comp_model)

        (data.mean, 
         data.conf_high, 
         data.conf_low) = extract_data(args, data_files)
        # data.mean = extract_data(args, data_files)

        all_data.append(data)

    return all_data


def get_data_files_from_splits(args, comp_model):
    data_files = []
    splits = load_json_list(args.train_test_splits)
    dir_path = os.path.join(args.load_dir, comp_model)

    for split_i in args.split_indexes:
        indexes = splits[split_i]
        train_indexes = splits[split_i]["train"]
        train_indexes = [x + 1 for x in train_indexes]
        train_indexes_str = '_'.join(str(x) for x in train_indexes)

        test_indexes = splits[split_i][args.data_type]
        test_indexes = [x + 1 for x in test_indexes]

        for partner_i, policy_i in enumerate(test_indexes):
            filename = f"{comp_model}_sad_{args.name_ext}{train_indexes_str}_vs_sad_{policy_i}.csv"
            filepath = os.path.join(dir_path, filename)
            if args.partner_indexes == "None" or \
                split_i < len(args.split_indexes) - 1 or \
                partner_i in args.partner_indexes:
                data_files.append(filepath)

    return data_files


def get_all_sad_data_files(args):
    data_files = []

    for i in args.sad_indexes:
        base_model = f"sad_{i + 1}"
        dir_path = os.path.join(args.sad_dir, base_model)

        for j in range(13):
            if i == j:
                continue
            filename = f"{base_model}_vs_sad_{j + 1}.csv"
            filepath = os.path.join(dir_path, filename)
            data_files.append(filepath)

    return data_files


def get_all_data_files(args, comp_model):
    dir_path = os.path.join(args.load_dir, comp_model)

    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    data_files = [os.path.join(dir_path, x) for x in onlyfiles if ".csv" in x]
    return data_files


def extract_data(args, data_files):
    # similarity_data = np.zeros
    similarity = np.zeros((args.datasize,2), dtype=int)

    for data_file in data_files:
        if not os.path.exists(data_file):
            print("not found:", data_file)
            continue
        data = genfromtxt(data_file, delimiter=',', dtype=int)
        print(data_file)
        similarity = np.add(similarity, data)

    # print(similarity)

    totals = np.sum(similarity, axis=1)
    pprint("totals")
    pprint(totals.tolist())
    pprint("similarity")
    pprint(similarity)

    mean = similarity[:,0] / totals

    #calculate 95% confidence interval with 56 successes in 100 trials
    # low, high = proportion_confint(count=56, nobs=100, method="wilson")

    # std = np.sqrt((similarity[:,0] * similarity[:,1]) / totals)
    # stderr = std / np.sqrt(totals)
    stderr = np.sqrt((similarity[:,0] * similarity[:,1]) / totals)

    # mean = np.mean(combined_data, axis=1)
    # std = np.std(combined_data, axis=1)
    # stderr = mean / np.sqrt(combined_data.shape[1])

    splits = load_json_list(args.train_test_splits)
    indexes = splits[0][args.data_type]

    num_samples = similarity.shape[0] * SAMPLES_PER_PAIR
    bin_conf = np.sqrt((mean * (1 - mean)) / num_samples)

    conf = bin_conf
    if args.data_type == "stderr":
        conf = stderr

    conf_high = mean + conf
    conf_low = mean - conf

    # print("conf:", conf.shape)
    # print(conf)
    # print("conf_high:", conf_high.shape)
    # print(conf_high)
    # print("conf_low:", conf_low.shape)
    # print(conf_low)

    return mean, conf_high, conf_low


def plot_data(args, all_data):
    time_steps = list(range(args.seq_start, args.seq_end))

    for data in all_data:
        mean = data.mean[args.seq_start:args.seq_end]
        plt.plot(time_steps, mean, label=data.name)
        conf_high = data.conf_high[args.seq_start:args.seq_end]
        conf_low = data.conf_low[args.seq_start:args.seq_end]
        plt.fill_between(time_steps, conf_high, conf_low, alpha=0.3)

    if len(args.split_indexes) == 1:
        args.title += f" - Split: {args.split_indexes[0]}"

    plt.legend(loc="best")
    plt.title(args.title)
    plt.xlabel("Time Step")
    plt.ylabel(args.ylabel)
    plt.show()


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, required=True)
    parser.add_argument("--sad_dir", type=str, default=None)
    parser.add_argument("--train_test_splits", type=str, default="None")
    parser.add_argument("--split_indexes", type=str, default="None")
    parser.add_argument("--partner_indexes", type=str, default="None")
    parser.add_argument("--compare_models", type=str, default="None")
    parser.add_argument("--seq_start", type=int, default=1)
    parser.add_argument("--seq_end", type=int, default=70)
    parser.add_argument("--datasize", type=int, default=80)
    parser.add_argument("--title", type=str, default="SAD Policy Differences")
    parser.add_argument("--ylabel", type=str, default="Similarity vs SAD")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--conf_type", type=str, default="bin_conf")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--sad_indexes", type=str, default="None")
    args = parser.parse_args()

    if args.compare_models == "None":
        args.compare_models = []
    else:
        args.compare_models = args.compare_models.split(",")

    if args.split_indexes == "None":
        args.split_indexes = []
    else:
        args.split_indexes = [int(x) for x in args.split_indexes.split(",")]

    if args.sad_indexes == "None":
        args.sad_indexes = []
    else:
        args.sad_indexes = [int(x) for x in args.sad_indexes.split(",")]

    if args.partner_indexes != "None":
        args.partner_indexes = [int(x) for x in args.partner_indexes.split(",")]

    if args.name_ext != "":
        args.name_ext += "_"

    args.load_dir = os.path.join("similarity_data", args.load_dir)
    if args.sad_dir is not None:
        args.sad_dir = os.path.join("similarity_data", args.sad_dir)

    plot_differences(args)

