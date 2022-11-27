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
import copy

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


def run_belief_cross_play(args):
    plot_data = belief_cross_play(args)
    pprint(plot_data)
    create_figures(plot_data, args)


def conv_str(conv):
    conv = conv[0]
    return f"{conv[0]}{conv[1]}"


def belief_cross_play(args):
    data = []
    xlabels = []
    ylabels = []
    min_value = float("inf")
    max_value = float("-inf")

    policy_conventions = load_json_list(args.policy_conventions)
    belief_conventions = load_json_list(args.belief_conventions)

    loaded_agents = load_agents(args)
    belief_model = load_belief_model(args)

    num_policies = len(policy_conventions)
    if args.policy_root != "None":
        num_policies = len(loaded_agents)

    for policy_index in range(num_policies):
        policy_name = create_policy_name(
                args, policy_index, policy_conventions)
        print("collecting data:", policy_name)

        belief_losses, xlabels, new_min, new_max = calculate_belief_loss_for_policy(
            args, loaded_agents, policy_index, 
            policy_conventions, belief_model, belief_conventions)

        min_value = min(min_value, new_min)
        max_value = max(max_value, new_max)

        ylabels.append(policy_name)
        data.append(belief_losses)

    if not args.auto_maxmin:
        min_value = args.colour_min
        max_value = args.colour_max

    return {
        "data": data,
        "xlabels": xlabels,
        "ylabels": ylabels,
        "min_value": min_value,
        "max_value": max_value,
    }


def load_agents(args):
    loaded_agents = []

    default_cfg = {
        "act_base_eps": 0.1,
        "act_eps_alpha": 7,
        "num_game_per_thread": 80,
        "num_player": 2,
        "train_bomb": 0,
        "max_len": 80,
        "sad": 1,
        "shuffle_color": 0,
        "hide_action": 0,
        "multi_step": 1,
        "gamma": 0.999,
        "parameterized": 0,
    }

    policy_list = [args.policy]
    if args.policy_root != "None":
        policy_list = common_utils.get_all_files(args.policy_root)

    for policy in policy_list:
        if args.sad_legacy:
            agent = utils.load_sad_model(policy, args.device)
            cfg = default_cfg
        else:
            agent, cfg = utils.load_agent(policy, {
                "vdn": False,
                "device": args.device,
                "uniform_priority": True,
            })

        if agent.boltzmann:
            boltzmann_beta = utils.generate_log_uniform(
                1 / cfg["max_t"], 1 / cfg["min_t"], cfg["num_t"]
            )
            boltzmann_t = [1 / b for b in boltzmann_beta]
        else:
            boltzmann_t = []

        loaded_agents.append((agent, cfg, boltzmann_t))

    return loaded_agents

def load_belief_model(args):
    belief_config = utils.get_train_config(args.belief_model)

    print("loading belief from: ", args.belief_model)
    return ARBeliefModel.load(
        args.belief_model,
        args.device,
        5,
        10,
        belief_config["fc_only"],
        belief_config["parameterized"],
        belief_config["num_parameters"],
        sad_legacy=args.sad_legacy,
    )

def create_policy_name(args, policy_index, policy_conventions):
    if args.policy_root != "None":
        return f"{policy_index + 1}"
    return conv_str(policy_conventions[policy_index])

def create_belief_name(args, belief_index, belief_conventions):
    if args.belief_num_parameters > 0:
        return f"{belief_index + 1}"
    return conv_str(belief_conventions[belief_index])

def calculate_belief_loss_for_policy(
        args, 
        loaded_agents, 
        policy_index, 
        policy_conventions,
        belief_model,
        belief_conventions,
):
    belief_losses = []
    xlabels = []
    min_value = float("inf")
    max_value = float("-inf")

    belief_model_for_runner = None
    if args.belief_runner:
        belief_model_for_runner = belief_model

    assert((args.num_game * 2) % args.batch_size == 0)
    num_batches = (int)((args.num_game * 2) / args.batch_size)

    num_belief_params = len(belief_conventions)
    if args.belief_num_parameters > 0:
        num_belief_params = args.belief_num_parameters

    agent = loaded_agents[0]
    if args.policy_root != "None":
        agent = loaded_agents[policy_index]

    convention_index = policy_index
    if args.policy_root != "None":
        convention_index = 0

    replay_buffer = generate_replay_data(
        agent,
        args.num_game, 
        args.seed, 
        0,
        num_thread=args.num_thread,
        device=args.device,
        convention=args.policy_conventions,
        convention_index=convention_index,
        belief_model=belief_model_for_runner,
        override=[args.override, args.override],
        sad_legacy=args.sad_legacy,
    )

    for belief_index in range(num_belief_params):
        for batch_index in range(num_batches):
            range_start = batch_index * args.batch_size
            range_end = batch_index * args.batch_size + args.batch_size
            sample_id_list = [*range(range_start, range_end, 1)]

            batch, _ = replay_buffer.sample_from_list(
                    args.batch_size, args.device, sample_id_list)

            values = []
            if args.loss_type == "per_card":
                loss = belief_model.loss_no_grad(batch, 
                    convention_index_override=belief_index)
                (_, avg_xent, _, _) = loss
                values.append(avg_xent.mean().item())

            elif args.loss_type == "response_playable":
                loss = belief_model.loss_response_playable(batch,
                    convention_index_override=belief_index)
                values.append(loss)

        value_mean = copy.copy(np.mean(values))

        min_value = min(min_value, value_mean)
        max_value = max(max_value, value_mean)

        belief_losses.append(value_mean)

        belief_name = create_belief_name(
                args, belief_index, belief_conventions)
        xlabels.append(belief_name)

    return belief_losses, xlabels, min_value, max_value

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
    sad_legacy=0,
):
    agent, cfgs, boltzmann_t = loaded_agent

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
                    sad_legacy, # sadLegacy
                    False, # beliefSadLegacy
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

        thread = hanalearn.HanabiThreadLoop(
                thread_games, thread_actors, True, t_idx)
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
    title = f"{args.loss_type}: sad vs pbelief_sad"
    ylabel = "Ground Truth Convention"
    xlabel = "Belief Convention"
    xticklabels = plot_data["xlabels"]
    yticklabels = plot_data["ylabels"]
    colour_max = plot_data["max_value"]
    colour_min = plot_data["min_value"]

    norm = mcolors.Normalize(vmin=colour_min, vmax=colour_max)
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
    parser.add_argument("--policy", type=str, default="None")
    parser.add_argument("--policy_root", type=str, default="None")
    parser.add_argument("--belief_model", type=str, required=True)
    parser.add_argument("--policy_conventions", default="None", type=str)
    parser.add_argument("--belief_conventions", default="None", type=str)
    parser.add_argument("--belief_num_parameters", default=0, type=int)
    parser.add_argument("--num_game", default=1000, type=int)
    parser.add_argument("--num_thread", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--loss_type", type=str, default="per_card")
    parser.add_argument("--belief_runner", type=int, default=0)
    parser.add_argument("--colour_max", type=float, default=1)
    parser.add_argument("--colour_min", type=float, default=0)
    parser.add_argument("--reverse_colour", type=float, default=0)
    parser.add_argument("--override", type=int, default=3)
    parser.add_argument("--sad_legacy", type=int, default=0)
    parser.add_argument("--auto_maxmin", type=int, default=0)

    args = parser.parse_args()

    run_belief_cross_play(args)
