import os
import sys
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import pprint
pprint = pprint.pprint


SPLIT_SIZES = {
    "sad": [
        "one", 
        "six", 
        "eleven"
    ],
    "iql": [
        "one", 
        "six", 
        "ten"
    ],
    "op": [
        # "one", 
        # "six", 
        # "ten",
    ]
}

SPLIT_TYPE_LABEL = {
    "one": "Small",
    "six": "Medium",
    "ten": "Large",
    "eleven": "Large",
}


def plot_action_stats(args):
    data, labels = read_data(args)
    plot_clustered(args, data, labels)
    # plot_stacked(args, data, labels)


def read_data(args):
    data = {
        "br": {
            "play": [],
            "discard": [],
            "hint_colour": [],
            "hint_rank": []
        },
        "sba": {
            "play": [],
            "discard": [],
            "hint_colour": [],
            "hint_rank": []
        }
    }

    files = [
        "br_sad_one",
        "br_sad_one",
    ]

    models = ["sad", "iql", "op"]
    model_types = ["br", "sba"]

    labels = []

    for model in models:
        for split_type in SPLIT_SIZES[model]:
            label = f"{model.upper()} {SPLIT_TYPE_LABEL[split_type]}"
            labels.append(label)
            for model_type in model_types:
                file_name = f"{model_type}_{model}_{split_type}.csv"
                file_path = os.path.join(args.folder, file_name)
                action_counts = np.genfromtxt(file_path, delimiter=',')

                total = action_counts[4]
                play = action_counts[0] / total
                discard = action_counts[1] / total
                hint_colour = action_counts[2] / total
                hint_rank = action_counts[3] /total

                data[model_type]["play"].append(play)
                data[model_type]["discard"].append(discard)
                data[model_type]["hint_colour"].append(hint_colour)
                data[model_type]["hint_rank"].append(hint_rank)

    return data, labels



def plot_clustered(args, data, labels):
    fig, ax = plt.subplots(1, 1)

    width = 0.4
    br_hint_colour = data["br"]["hint_colour"]
    sba_hint_colour = data["sba"]["hint_colour"]
    colours = ["#d62728", "#1f77b4"]

    values = np.arange(len(labels))

    ax.bar(values - width/2, br_hint_colour, width, label="BR", color=colours[0])
    ax.bar(values + width/2, sba_hint_colour, width, label="SBA", color=colours[1])

    ax.set_ylabel("Hint Colour")
    yticks = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])
    labels.insert(0, "")
    ax.set_xticklabels(labels)

    ax.legend(loc="upper right")
    # ax.legend(bbox_to_anchor=(0.7, 1))

    plt.show()


def plot_stacked(args, data, labels):
    fig, ax = plt.subplots(1, 1)

    width = 0.4
    actions = ["hint_colour", "hint_rank", "discard", "play"]
    values = np.arange(len(labels))
    colours = ["#d62728", "#1f77b4", "#EDB732", "#2E9B4A"]

    prev_br_data = [0] * len(data["br"]["play"])
    prev_sba_data = [0] * len(data["sba"]["play"])

    for i, action in enumerate(actions):
        br_data = data["br"][action]
        sba_data = data["sba"][action]

        ax.bar(values - width/2 - 0.02, br_data, width, label=action,
                color=colours[i], bottom=prev_br_data)
        ax.bar(values + width/2 + 0.02, sba_data, width, 
                color=colours[i], bottom=prev_sba_data)

        prev_br_data = [ sum(x) for x in zip(br_data, prev_br_data)]
        prev_sba_data = [ sum(x) for x in zip(sba_data, prev_sba_data)]

    ax.set_ylabel("Percentage")
    yticks = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

    labels = [ f"BR   SBA\n{x}" for x in labels ]
    labels.insert(0, "")
    ax.set_xticklabels(labels)

    ax.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="action_stats")
    args = parser.parse_args()
    plot_action_stats(args)
