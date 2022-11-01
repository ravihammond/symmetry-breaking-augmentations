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

from eval_belief import *
import common_utils
import utils


def run_belief_xent_cross_play(args):
    plot_data = belief_xent_cross_play(args)
    create_figures(plot_data)


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def conv_str(conv):
    conv = conv[0]
    return f"{conv[0]}{conv[1]}"


def belief_xent_cross_play(args):
    data = []
    labels = []

    conventions = load_json_list(args.convention)

    loaded_agent = utils.load_agent(args.policy, {
        "vdn": False,
        "device": args.device,
        "uniform_priority": True,
    })

    belief_config = utils.get_train_config(args.belief_model)

    print("loading file from: ", args.belief_model)
    belief_model = ARBeliefModel.load(
        args.belief_model,
        args.device,
        5,
        10,
        belief_config["fc_only"],
        belief_config["parameterized"],
        belief_config["num_conventions"],
    )

    for act_convention_index in range(len(conventions)):
        act_convention = conv_str(conventions[act_convention_index])
        print(f"collecting data: {act_convention}")
        batch = generate_replay_data(
            loaded_agent,
            args.num_game, 
            args.seed, 
            0,
            convention=args.convention,
            convention_index=act_convention_index,
            override=[3, 3],
        )
        row = []

        for belief_convention_index in range(len(conventions)):
            result = belief_model.loss_no_grad(batch, 
                convention_index_override=belief_convention_index)
            (xent, avg_xent, avg_xent_v0, nll_per_card) = result

            avg_xent = avg_xent.mean().item()

            row.append(avg_xent)

        labels.append(f"{act_convention}")
        data.append(row)

    return {
        "data": data,
        "labels": labels,
    }

def create_figures(plot_data, colour_max=3):
    print("creating figures")
    data = plot_data["data"]
    title = "XEntropy: obl1f vs pbelief1_obl1f"
    ylabel = "Ground Truth Convention"
    xlabel = "Belief Convention"
    xticklabels = plot_data["labels"]
    yticklabels = plot_data["labels"]

    norm = mcolors.Normalize(vmin=0., vmax=colour_max)
    # see note above: this makes all pcolormesh calls consistent:
    pc_kwargs = {'rasterized': True, 'cmap': 'cividis_r', 'norm': norm}
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
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--belief_model", type=str, required=True)
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--num_game", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    run_belief_xent_cross_play(args)

