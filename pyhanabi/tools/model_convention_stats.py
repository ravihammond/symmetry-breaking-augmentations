import argparse
import os
import sys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import re
import pprint
pprint = pprint.pprint

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
from model_zoo import model_zoo

COLOURS = "RYGWB"
RANKS = "12345"
CARDS = "01234"

ACTION_MAP = {
    "C": COLOURS,
    "R": RANKS,
    "P": CARDS,
    "D": CARDS
}


def convention_data(args):
    weight_files = load_weights(args)
    _, actors = run_evaluation(args, weight_files)

    stats = extract_convention_stats(actors, args.actor)
    pprint(stats)

    plot_data = generate_plot_data(stats, args)
    convention_matrix(*plot_data, title=args.title, colour_max=args.colour_max)


def load_weights(args):
    weight_files = []
    if args.num_player == 2:
        if args.weight2 is None:
            args.weight2 = args.weight1
        weight_files = [args.weight1, args.weight2]
    elif args.num_player == 3:
        if args.weight2 is None:
            weight_files = [args.weight1 for _ in range(args.num_player)]
        else:
            weight_files = [args.weight1, args.weight2, args.weight3]

    for i, wf in enumerate(weight_files):
        if wf in model_zoo:
            weight_files[i] = model_zoo[wf]

    assert len(weight_files) == 2
    return weight_files


def run_evaluation(args, weight_files):
    score, perfect, _, scores, actors = evaluate_saved_model(
        weight_files,
        args.num_game,
        args.seed,
        args.bomb,
        num_run=args.num_run,
        device=args.device,
        convention=args.convention,
        convention_sender=args.convention_sender,
        override=[args.override0, args.override1],
    )

    return scores, actors


def extract_convention_stats(actors, actor):
    stats = {}
    signal_response_data(stats, "D", CARDS, actors, actor)
    signal_response_data(stats, "P", CARDS, actors, actor)
    signal_response_data(stats, "C", COLOURS, actors, actor)
    signal_response_data(stats, "R", RANKS, actors, actor)
    return stats


def signal_response_data(stats, signal_type, type_map, actors, actor):
    for s_idx in range(5):
        signal = f"{signal_type}{type_map[s_idx]}"
        total = sum_stats(signal, actors, actor)

        for r_idx in range(5):
            response_data(stats, signal, "D", r_idx, CARDS, actors, total, actor)
            response_data(stats, signal, "P", r_idx, CARDS, actors, total, actor)
            response_data(stats, signal, "C", r_idx, COLOURS, actors, total, actor)
            response_data(stats, signal, "R", r_idx, RANKS, actors, total, actor)


def response_data(
        stats, signal, response_type, index, type_map, actors, total, actor):
    response = f"{signal}_{response_type}{type_map[index]}"
    response_sum = sum_stats(response, actors, actor)
    stats[response] = divide(response_sum, total)
    return response_sum


def sum_stats(key, actors, actor):
    stats = []
    for i, g in enumerate(actors): 
        if actor == -1 or i % 2 == actor:
            if key in g.get_stats():
                stats.append(g.get_stats()[key])
    return int(sum(stats))

def divide(n, total):
    if total == 0:
        return 0
    return (n / total)

def generate_plot_data(stats, args):
    xticklabels = ticklabels(args.response_actions)
    yticklabels = ticklabels(args.signal_actions)

    plot_data = []

    for yticklabel in yticklabels:
        row_data = []
        for xticklabel in xticklabels:
            row_data.append(stats[f"{yticklabel}_{xticklabel}"])
        plot_data.append(row_data)

    return plot_data, xticklabels, yticklabels

def ticklabels(actions):
    tick_labels = []
    for action_type in actions:
        for signal_move in ACTION_MAP[action_type]:
            tick_labels.append(f"{action_type}{signal_move}")

    return tick_labels

def convention_matrix(data, xticklabels, yticklabels, colour_max=1.0, 
        xlabel="response t+1", ylabel="signal t", title="None"):
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
    parser.add_argument("--weight1", default=None, type=str, required=True)
    parser.add_argument("--weight2", default=None, type=str)
    parser.add_argument("--weight3", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bomb", default=0, type=int)
    parser.add_argument("--num_game", default=5000, type=int)
    parser.add_argument("--num_run", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--convention_sender", default=0, type=int)
    parser.add_argument("--override0", default=0, type=int)
    parser.add_argument("--override1", default=0, type=int)
    parser.add_argument("--signal_actions", default="CR", type=str)
    parser.add_argument("--response_actions", default="DP", type=str)
    parser.add_argument("--title", default="None", type=str)
    parser.add_argument("--colour_max", default=0.7, type=float)
    parser.add_argument("--actor", default=-1, type=int)
    args = parser.parse_args()

    convention_data(args)

