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


lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model, load_agents
import common_utils
import utils
from tools import model_zoo


def run_cross_play(args):
    models = common_utils.get_all_files(args.root, "model0.pthw")
    if args.include is not None:
        models = filter_include(models, args.include)
    if args.exclude is not None:
        models = filter_exclude(models, args.exclude)

    models = sorted(models, key=str.lower)

    pprint(models)

    if args.convention == "None":
        data = cross_play(models, args, 1)
    else:
        if len(models) > 1:
            sys.exit("model length needs to be 1 when testing conventions")
        data = cross_play_conventions(models[0], args, 1)

    create_figures(data)


def filter_include(entries, includes):
    if not isinstance(includes, list):
        includes = [includes]

    keep = []
    for entry in entries:
        for include in includes:
            if include in entry:
                keep.append(entry)
    return keep


def filter_exclude(entries, excludes):
    if not isinstance(excludes, list):
        excludes = [excludes]
    keep = []
    for entry in entries:
        for exclude in excludes:
            if exclude in entry:
                break
        else:
            keep.append(entry)
    return keep


def cross_play(models, args, seed):
    combs = list(itertools.combinations_with_replacement(models, args.num_player))
    perfs = defaultdict(list)

    for comb in combs:
        num_model = len(set(comb))
        score = evaluate_saved_model(comb, args.num_game, seed, 0)[0]
        perfs[num_model].append(score)

    for num_model, scores in perfs.items():
        print(
            f"#model: {num_model}, #groups {len(scores)}, "
            f"score: {np.mean(scores):.2f} "
            f"+/- {np.std(scores) / np.sqrt(len(scores) - 1):.2f}"
        )


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def conv_str(conv):
    conv = conv[0]
    return f"{conv[0]}{conv[1]}"


def cross_play_conventions(model, args, seed):
    all_scores = []
    labels = []

    conventions = load_json_list(args.convention)
    model_name = os.path.basename(os.path.dirname(model))

    pre_loaded_data = load_agents([model, model])

    for i in range(len(conventions)):
        row = []
        for j in range(len(conventions)):
            model_key = f"{conv_str(conventions[i])}" + \
                        f":{conv_str(conventions[j])}" 
            print("evaluting:", model_key)
            scores = evaluate_saved_model(
                    [model, model], 
                    args.num_game, 
                    seed, 
                    0,
                    convention=args.convention,
                    override=[3, 3],
                    convention_indexes=[i, j],
                    pre_loaded_data=pre_loaded_data,
            )[0]
            row.append(np.mean(scores))
        labels.append(f"{conv_str(conventions[i])}")
        all_scores.append(row)

    return {
        "scores": all_scores,
        "labels": labels,
    }

def create_figures(plot_data, colour_max=25):
    data = plot_data["scores"]
    title = "OBL1f All Conventions"
    ylabel = "Player 1 Agent"
    xlabel = "Player 2 Agent"
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
    parser.add_argument("--root", default=None, type=str)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--include", default=None, type=str, nargs="+")
    parser.add_argument("--exclude", default=None, type=str, nargs="+")
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--num_game", default=1000, type=int)

    args = parser.parse_args()

    run_cross_play(args)

