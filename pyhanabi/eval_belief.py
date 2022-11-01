# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os 
import sys
import argparse
import pprint
import pickle
import json
import numpy as np
import torch
import pprint
pprint = pprint.pprint

from create import *
import rela
import r2d2 
import utils 
from eval import load_agents
from belief_model import ARBeliefModel


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
    belief_stats=False,
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
                    False, # beliefStats
                )
                actors.append(actor)
                all_actors.append(actor)
            thread_actors.append(actors)
            thread_games.append(games[g_idx])
            seed += 1
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    runner.start()

    context.start()
    context.join()
    runner.stop()

    batch, _ = replay_buffer.sample(replay_buffer_size, device)

    return batch


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)
