import argparse
import os
import sys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import re
import json 
import pprint
pprint = pprint.pprint

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
from model_zoo import model_zoo
from calculate_convention_stats import extract_convention_stats

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
    conventions = load_convention_strings(args.convention)
    weight_files = load_weights(args)
    _, actors = run_evaluation(args, weight_files, conventions)

    stats = extract_convention_stats(actors, args, conventions)

    model_name = os.path.basename(os.path.dirname(args.weight1))
    create_figures(stats, conventions, model_name, args)
    pprint(stats)
    # create_plots(stats, conventions)

def load_convention_strings(convention_path):
    if convention_path == "None":
        return []

    convention_file = open(convention_path)
    conventions = json.load(convention_file)

    convention_strings = []

    for convention in conventions:
        convention_str = ""
        for i, two_step in enumerate(convention):
            if i > 0:
                convention_str + '-'
            convention_str += two_step[0] + two_step[1]
        convention_strings.append(convention_str)

    return convention_strings

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


def run_evaluation(args, weight_files, conventions):
    num_game_multiplier = 1 if len(conventions) == 0 else len(conventions)

    score, perfect, _, scores, actors = evaluate_saved_model(
        weight_files,
        args.num_game * num_game_multiplier,
        args.seed,
        args.bomb,
        num_run=args.num_run,
        device=args.device,
        convention=args.convention,
        override=[args.override0, args.override1],
    )

    return scores, actors

def create_figures(stats, conventions, title, args):
    figx = 5 * (2 if args.split else 1)
    figy = 5 * (1 if len(conventions) == 0 else len(conventions) + 1)
    fig = plt.figure(constrained_layout=True, figsize=(figx, figy))

    rows = len(conventions) + 1
    subfigs = fig.subfigures(rows, 1)
    if rows == 1:
        subfigs = [subfigs]

    for i in range(rows):
        create_figure_row(subfigs[i], stats[i], args)

    fig.suptitle(title, fontsize='xx-large')

    plt.show()

def create_figure_row(fig, stats, args):
    plots = stats["plots"]
    title = stats["title"]
    axs = fig.subplots(1, len(plots), sharey=True)
    if len(plots) == 1:
        axs = [axs]

    fig.suptitle(title)

    for i, plot_stats in enumerate(plots):
        plot_data = generate_plot_data(plot_stats, args, title, i)
        create_plot(axs[i], *plot_data, colour_max=args.colour_max)


def generate_plot_data(stats, args, convention, player_idx):
    prefix = convention + ":"
    if convention == "All":
        prefix = ""
    
    xticklabels = ticklabels(args.response_actions)
    yticklabels = ticklabels(args.signal_actions)

    plot_data = []

    for yticklabel in yticklabels:
        row_data = []
        for xticklabel in xticklabels:
            row_data.append(stats[f"{prefix}{player_idx}:{yticklabel}_{xticklabel}"])
        plot_data.append(row_data)

    return plot_data, xticklabels, yticklabels


def ticklabels(actions):
    tick_labels = []
    for action_type in actions:
        for signal_move in ACTION_MAP[action_type]:
            tick_labels.append(f"{action_type}{signal_move}")

    return tick_labels


def create_plot(ax, data, xticklabels, yticklabels, colour_max=1.0):
    norm = mcolors.Normalize(vmin=0., vmax=colour_max)
    pc_kwargs = {'rasterized': True, 'cmap': 'cividis', 'norm': norm}

    im = ax.imshow(data, **pc_kwargs)

    ax.set_xlabel("response t+1")
    ax.set_ylabel("signal t")

    ax.set_xticks(range(len(xticklabels)));
    ax.set_xticklabels(xticklabels);
    ax.xaxis.tick_top()
    ax.tick_params('both', length=1, width=1, which='major')

    ax.set_yticks(range(len(yticklabels)));
    ax.set_yticklabels(yticklabels);


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight1", default=None, type=str, required=True)
    parser.add_argument("--weight2", default=None, type=str)
    parser.add_argument("--weight3", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bomb", default=0, type=int)
    parser.add_argument("--num_game", default=1000, type=int)
    parser.add_argument("--num_run", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--override0", default=0, type=int)
    parser.add_argument("--override1", default=0, type=int)
    parser.add_argument("--signal_actions", default="DPCR", type=str)
    parser.add_argument("--response_actions", default="DPCR", type=str)
    parser.add_argument("--title", default="None", type=str)
    parser.add_argument("--colour_max", default=0.7, type=float)
    parser.add_argument("--split", default=1, type=int)
    args = parser.parse_args()

    convention_data(args)

