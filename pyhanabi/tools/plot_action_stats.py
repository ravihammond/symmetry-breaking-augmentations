import os
import sys
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pprint
pprint = pprint.pprint


def plot_action_stats(args):
    df = pd.read_csv(args.file)
    df = convert_data(args, df)
    plot_data(args, df)


def convert_data(args, df):
    df["model_type"] = df.model.str.extract(r"(^br|^sba)")
    df["hint_colour"] = df["action_is_hint_colour_mean"]
    df["hint_rank"] = df["action_is_hint_rank_mean"]
    df["discard"] = df["action_is_discard_mean"]
    df["play"] = df["action_is_play_mean"]

    df = df.query("model_type in ['br', 'sba']")

    df_grouped = df.groupby(["model_type", "turn"]).mean().reset_index()
    df_grouped["hint_colour_mean"] = df_grouped["hint_colour"]
    df_grouped["hint_rank_mean"] = df_grouped["hint_rank"]
    df_grouped["discard_mean"] = df_grouped["discard"]
    df_grouped["play_mean"] = df_grouped["play"]

    df_sem = df.groupby(["model_type", "turn"]).sem().reset_index()
    df_grouped["hint_colour_sem"] = df_sem["hint_colour"]
    df_grouped["hint_rank_sem"] = df_sem["hint_rank"]
    df_grouped["discard_sem"] = df_sem["discard"]
    df_grouped["play_sem"] = df_sem["play"]

    df_grouped = df_grouped[df_grouped["turn"] >= args.turn_start]
    df_grouped = df_grouped[df_grouped["turn"] <= args.turn_end - 1]
    df_grouped = df_grouped.reset_index()

    df_grouped = df_grouped[[ 
        "model_type", 
        "turn", 
        "hint_colour_mean",
        "hint_colour_sem",
        "hint_rank_mean",
        "hint_rank_sem",
        "discard_mean",
        "discard_sem",
        "play_mean",
        "play_sem"
    ]]
    
    return df_grouped


def plot_data(args, df):
    fig, axes = plt.subplots(2, 2, sharey=True,
            constrained_layout=True, figsize=(6, 6))

    titles = ["a)", "b)", "c)", "d)"]
    data_types = ["hint_colour", "hint_rank", "discard", "play"]
    y_labels = [
        "Hint Colour", 
        "Hint Rank", 
        "Discard", 
        "Play"
    ]

    for i, split_type in enumerate(data_types):
        create_plot(
            args,
            axes[i // 2][i % 2], 
            df,
            data_types[i],
            titles[i],
            y_labels[i],
            i == 0,
        )

    plt.show()


def create_plot(args, ax, df, data_type, title, y_label, show_legend):
    models = ["br", "sba"]
    colours = ["#d62728", "#1f77b4"]
    mean_type = f"{data_type}_mean"
    sem_type = f"{data_type}_sem"

    x_vals = np.arange(args.turn_start, args.turn_end)

    for i, model in enumerate(models):
        y_mean = df[df["model_type"] == model][mean_type]
        y_sem = df[df["model_type"] == model][sem_type]
        ax.plot(x_vals, y_mean, colours[i], label=model.upper())
        ax.fill_between(x_vals, y_mean + y_sem, y_mean - y_sem,
                        color=colours[i], alpha=0.3)
    if show_legend:
        ax.legend(loc="best")
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Turn")
    ax.set_ylabel(y_label)
    ax.set_ylim(0,0.6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--turn_start", type=int, default=1)
    parser.add_argument("--turn_end", type=int, default=70)
    args = parser.parse_args()
    plot_action_stats(args)
