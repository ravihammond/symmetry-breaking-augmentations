# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import time
import os 
import sys
import argparse
import pprint
import pickle
import json
import numpy as np
import itertools
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pprint
pprint = pprint.pprint

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

from create import *
import rela
import r2d2 
import utils 
import common_utils
from eval_belief import *
from eval import load_agents
from belief_model import ARBeliefModel


def run_belief_xent_cross_play(args):
    plot_data = belief_xent_cross_play(args)
    pprint(plot_data)
    print
    create_figures(plot_data, args)


def conv_str(conv):
    conv = conv[0]
    return f"{conv[0]}{conv[1]}"


def belief_xent_cross_play(args):
    data = []
    xlabels = []
    ylabels = []

    policy_conventions = load_json_list(args.policy_conventions)
    belief_conventions = load_json_list(args.belief_conventions)

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
    belief_model_for_runner = None
    if args.belief_runner:
        belief_model_for_runner = belief_model

    assert((args.num_game * 2) % args.batch_size == 0)
    num_batches = (int)((args.num_game * 2) / args.batch_size)

    for i, policy_convention_index in enumerate(range(len(policy_conventions))):
        policy_convention_str = conv_str(policy_conventions[policy_convention_index])
        print(f"collecting data: {policy_convention_str}")
        replay_buffer = generate_replay_data(
            loaded_agent,
            args.num_game, 
            args.seed, 
            0,
            device=args.device,
            convention=args.policy_conventions,
            convention_index=policy_convention_index,
            belief_model=belief_model_for_runner,
            override=[args.override, args.override],
        )
        row = []

        for belief_convention_index in range(len(belief_conventions)):
            for batch_index in range(num_batches):
                range_start = batch_index * args.batch_size
                range_end = batch_index * args.batch_size + args.batch_size
                sample_id_list = [*range(range_start, range_end, 1)]

                batch, _ = replay_buffer.sample_from_list(
                        args.batch_size, args.device, sample_id_list)

                values = []
                if args.loss_type == "per_card":
                    loss = belief_model.loss_no_grad(batch, 
                        convention_index_override=belief_convention_index)
                    (_, avg_xent, _, _) = loss
                    values.append(avg_xent.mean().item())

                elif args.loss_type == "response_playable":
                    loss = belief_model.loss_response_playable(batch,
                        convention_index_override=belief_convention_index)
                    values.append(loss)

            row.append(np.mean(values))

            if i == 0:
                belief_convention_str = conv_str(
                        belief_conventions[belief_convention_index])
                xlabels.append(belief_convention_str)

        ylabels.append(policy_convention_str)
        data.append(row)

    return {
        "data": data,
        "xlabels": xlabels,
        "ylabels": ylabels,
    }


def generate_replay_data(
    loaded_agent,
    num_game,
    seed,
    bomb,
    *,
    num_thread=10,
    max_len=80,
    overwrite=None,
    device="cuda:0",
    convention="None",
    convention_index=None,
    override=[0, 0],
    belief_model=None,
    num_player=2,
):
    agent, cfgs = loaded_agent

    boltzmann_beta = utils.generate_log_uniform(
        1 / cfgs["max_t"], 1 / cfgs["min_t"], cfgs["num_t"]
    )
    boltzmann_t = [1 / b for b in boltzmann_beta]

    if num_game < num_thread:
        num_thread = num_game

    runner = rela.BatchRunner(agent.clone(device), device)
    runner.add_method("act", 5000)
    runner.add_method("compute_priority", 100)

    belief_runner = None
    belief_stats = False
    if belief_model is not None:
        belief_runner = rela.BatchRunner(
                belief_model, belief_model.device, 5000, ["sample"])
        belief_stats = True

    context = rela.Context()
    threads = []

    games = create_envs(
        num_game,
        seed,
        cfgs["num_player"],
        cfgs["train_bomb"],
        cfgs["max_len"]
    )

    replay_buffer_size = num_game * 2

    replay_buffer = rela.RNNPrioritizedReplay(
        replay_buffer_size,
        seed,
        1.0,  # priority exponent
        0.0,  # priority weight
        3, #prefetch
    )

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []

    partner_idx = 0

    conventions = load_json_list(convention)

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                actor = hanalearn.R2D2Actor(
                    runner, # runner
                    seed, # seed
                    num_player, # numPlayer
                    i, # playerIdx
                    [0], # epsList
                    [], # tempList
                    False, # vdn
                    cfgs["sad"], # sad
                    False, # shuffleColor
                    cfgs["hide_action"], # hideAction
                    False, # trinary
                    replay_buffer, # replayBuffer
                    cfgs["multi_step"], # multiStep
                    cfgs["max_len"], # seqLen
                    cfgs["gamma"], # gamma
                    conventions, # convention
                    cfgs["parameterized"], # actParameterized
                    convention_index, # conventionIdx
                    override[i], # conventionOverride
                    False, # fictitiousOverride
                    True, # useExperience
                    belief_stats, # beliefStats
                )

                if belief_stats:
                    actor.set_belief_runner_stats(belief_runner)

                actors.append(actor)
                all_actors.append(actor)

            for i in range(num_player):
                partners = actors[:]
                partners[i] = None
                actors[i].set_partners(partners)

            thread_actors.append(actors)
            thread_games.append(games[g_idx])
            seed += 1

        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    runner.start()

    if belief_runner is not None:
        belief_runner.start()

    context.start()
    context.join()

    runner.stop()

    if belief_runner is not None:
        belief_runner.stop()

    return replay_buffer


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)

def create_figures(plot_data, args):
    print("creating figures")
    data = plot_data["data"]
    title = f"{args.loss_type}: obl1f_all_colours vs pbelief1_obl1f_all_colours"
    ylabel = "Ground Truth Convention"
    xlabel = "Belief Convention"
    xticklabels = plot_data["xlabels"]
    yticklabels = plot_data["ylabels"]

    norm = mcolors.Normalize(vmin=args.colour_min, vmax=args.colour_max)
    # see note above: this makes all pcolormesh calls consistent:
    colour_map = "cividis_r" if args.reverse_colour else "cividis"
    pc_kwargs = {'rasterized': True, 'cmap': colour_map, 'norm': norm}
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
    parser.add_argument("--policy_conventions", default="None", type=str)
    parser.add_argument("--belief_conventions", default="None", type=str)
    parser.add_argument("--num_game", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--loss_type", type=str, default="per_card")
    parser.add_argument("--belief_runner", type=int, default=0)
    parser.add_argument("--colour_max", type=float, default=1)
    parser.add_argument("--colour_min", type=float, default=0)
    parser.add_argument("--reverse_colour", type=float, default=0)
    parser.add_argument("--override", type=int, default=3)

    args = parser.parse_args()

    run_belief_xent_cross_play(args)
