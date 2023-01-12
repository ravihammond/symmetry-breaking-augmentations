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
        else:
            data_files = get_all_data_files(args, comp_model)

        (data.mean, 
         data.stderr_high, 
         data.stderr_low) = extract_data(data_files)

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

        test_indexes = splits[split_i]["test"]
        test_indexes = [x + 1 for x in test_indexes]

        for policy_i in test_indexes:
            filename = f"{comp_model}_sad_{train_indexes_str}_vs_sad_{policy_i}.csv"
            filepath = os.path.join(dir_path, filename)
            data_files.append(filepath)

    return data_files


def get_all_data_files(args, comp_model):
    dir_path = os.path.join(args.load_dir, comp_model)

    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    data_files = [os.path.join(dir_path, x) for x in onlyfiles if ".csv" in x]
    return data_files


def extract_data(data_files):
    # similarity_data = np.zeros
    combined_data = np.zeros((args.datasize,0))

    for data_file in data_files:
        data = genfromtxt(data_file, delimiter=',')
        data = np.expand_dims(data, axis=1)
        combined_data = np.hstack((combined_data, data))

    mean = np.mean(combined_data, axis=1)
    std = np.std(combined_data, axis=1)
    stderr = mean / np.sqrt(combined_data.shape[1])
    stderr_high = mean + stderr
    stderr_low = mean - stderr

    return mean, stderr_high, stderr_low


def plot_data(args, all_data):
    time_steps = list(range(args.seq_end - args.seq_start))

    for data in all_data:
        plt.plot(time_steps, data.mean, label=data.name)
        plt.fill_between(time_steps, data.stderr_high, data.stderr_low, alpha=0.1)

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
    parser.add_argument("--train_test_splits", type=str, default="None")
    parser.add_argument("--split_indexes", type=str, default="None")
    parser.add_argument("--compare_models", type=str, default="None")
    parser.add_argument("--seq_start", type=int, default=0)
    parser.add_argument("--seq_end", type=int, default=80)
    parser.add_argument("--datasize", type=int, default=80)
    parser.add_argument("--title", type=str, default="SAD Policy Differences")
    parser.add_argument("--ylabel", type=str, default="Similarity vs SAD")
    args = parser.parse_args()

    if args.compare_models == "None":
        args.compare_models = []
    else:
        args.compare_models = args.compare_models.split(",")

    if args.split_indexes == "None":
        args.split_indexes = []
    else:
        args.split_indexes = [int(x) for x in args.split_indexes.split(",")]

    args.load_dir = os.path.join("similarity_data", args.load_dir)

    plot_differences(args)

