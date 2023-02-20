# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import json
import pprint
pprint = pprint.pprint
import itertools
from collections import defaultdict
import numpy as np
import matplotlib.colors as mcolors 
import matplotlib.pyplot as plt
import copy
import re


lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model, load_agents
import common_utils
import utils
from tools import model_zoo

def run_cross_play(args):
    margs = extract_model_args(args)

    models = []

    for marg in margs:
        if marg["all_models"]:
            models.append(common_utils.get_all_files(marg["root"]))
        else:
            models.append(common_utils.get_all_files_with_extention(
                marg["root"], args.model_epoch))

    data = cross_play(models, args, margs)

    create_figure(data)


def extract_model_args(args):
    model_args1 = {
        "root": args.root1,
        "all_models": args.all_models1,
        "convention": args.convention1,
        "num_parameters": args.num_parameters1,
        "override": args.override1,
        "sad_legacy": args.sad_legacy1,
    }

    if args.root2 == "None":
        return [model_args1, model_args1]

    model_args2 = {
        "root": args.root2,
        "all_models": args.all_models2,
        "convention": args.convention2,
        "num_parameters": args.num_parameters2,
        "override": args.override2,
        "sad_legacy": args.sad_legacy2,
    }

    return [model_args1, model_args2]


def cross_play(all_models, args, margs):
    all_scores = []
    labels = []

    conventions = [ 
        load_json_list(margs[i]["convention"]) 
        for i in range(args.num_player) 
    ]

    num_parameters = [ 
        len(conventions[i]) 
        if len(conventions[i]) > 0 else 
        margs[i]["num_parameters"]
        for i in range(args.num_player)
    ]

    num_parameters = []
    for i in range(args.num_player):
        if len(all_models[i]) > 1:
            num_parameters.append(len(all_models[i]))
            continue
        if len(conventions[i]) > 0:
            num_parameters.append(len(conventions[i]))
            continue
        if margs[i]["num_parameters"] > 0:
            num_parameters.append(margs[i]["num_parameters"])
            continue
    assert(len(num_parameters) == args.num_player)

    sad_legacy = [margs[i]["sad_legacy"] for i in range(2)]

    for i in range(num_parameters[0]):
        row = []
        for j in range(num_parameters[1]):
            print(f"evaluating: obl_{i + 1}, obl_{j + 1}")

            models = []
            model_idx = [i, j]
            for k in range(args.num_player):
                if margs[k]["num_parameters"] > 0:
                    models.append(all_models[k][0])
                else:
                    models.append(all_models[k][model_idx[k]])

            convention_indexes = [i, j]

            scores = evaluate_saved_model(
                    models,
                    args.num_game, 
                    args.seed, 
                    0,
                    convention=args.convention1,
                    override=[args.override1, args.override2],
                    convention_indexes=convention_indexes,
                    device=args.device,
                    sad_legacy=sad_legacy,
            )[0]

            value_mean = copy.copy(np.mean(scores))
            row.append(value_mean)

        labels.append(f"{i + 1}")
        all_scores.append(row)

    return {
        "scores": all_scores,
        "labels": labels,
    }


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def conv_str(conv):
    conv = conv[0]
    return f"{conv[0]}{conv[1]}"


def create_model_key(i, conventions):
    if len(conventions) == 0:
        return f"{i + 1}"
    return conv_str(conventions[i])


def create_figure(plot_data, colour_max=25):
    data = plot_data["scores"]
    title = "XP SAD vs POBL1_SAD"
    ylabel = "Player 1 Agent (SAD)"
    xlabel = "Player 2 Agent (POBL1_SAD)"
    xticklabels = plot_data["labels"]
    yticklabels = plot_data["labels"]

    norm = mcolors.Normalize(vmin=0., vmax=colour_max)
    # see note above: this makes all pcolormesh calls consistent:
    pc_kwargs = {'rasterized': True, 'cmap': 'cividis', 'norm': norm}
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(data, **pc_kwargs)
    cb = fig.colorbar(im, ax=ax,fraction=0.024, pad=0.04)
    cb.ax.tick_params(length=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(xticklabels)));
    ax.set_xticklabels(xticklabels);
    ax.xaxis.tick_top()
    ax.tick_params('both', length=1, width=1, which='major')
    plt.yticks(range(len(yticklabels)));
    ax.set_yticklabels(yticklabels);
    if title != "None":
        plt.title(title)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_game", default=1000, type=int)
    parser.add_argument("--model_epoch", default="model0.pthw", type=str)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--root1", default="None", type=str, required=True)
    parser.add_argument("--root2", default="None", type=str)
    parser.add_argument("--all_models1", default=0, type=int)
    parser.add_argument("--all_models2", default=0, type=int)
    parser.add_argument("--convention1", default="None", type=str)
    parser.add_argument("--convention2", default="None", type=str)
    parser.add_argument("--num_parameters1", default=0, type=int)
    parser.add_argument("--num_parameters2", default=0, type=int)
    parser.add_argument("--override1", default=0, type=int)
    parser.add_argument("--override2", default=0, type=int)
    parser.add_argument("--sad_legacy1", default=0, type=int)
    parser.add_argument("--sad_legacy2", default=0, type=int)
    parser.add_argument("--save_values", default=0, type=int)

    args = parser.parse_args()

    run_cross_play(args)

