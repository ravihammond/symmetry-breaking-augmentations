# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
import json
import numpy as np
import torch
import sys
import random
import pprint
pprint = pprint.pprint

from create import *
import rela
import r2d2 
import utils
import gc

def evaluate(
    agents,
    num_game,
    seed,
    bomb,
    eps,
    sad,
    hide_action,
    *,
    num_thread=10,
    max_len=80,
    device="cuda:0",
    convention=[],
    override=[0, 0],
    act_parameterized=[0, 0],
    belief_stats=False,
    belief_model_path="None",
):
    """
    evaluate agents as long as they have a "act" function
    """
    if num_game < num_thread:
        num_thread = num_game

    num_player = len(agents) 
    if not isinstance(hide_action, list):
        hide_action = [hide_action for _ in range(num_player)]
    if not isinstance(sad, list):
        sad = [sad for _ in range(num_player)]

    # Create Batch Runners only if agent is a learned r2d2 agent.
    runners = [rela.BatchRunner(agent.clone(device), device, 1000, ["act"])
            for agent in agents]

    belief_runner = None
    if belief_stats:
        belief_runner = create_belief_runner(belief_model_path, device)


    context = rela.Context()
    threads = []

    games = create_envs(num_game, seed, num_player, bomb, max_len)

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            convention_index = 0
            if len(convention) > 0:
                convention_index = random.randint(0, len(convention) - 1)
            for i in range(num_player):
                actor = hanalearn.R2D2Actor(
                    runners[i], # runner
                    num_player, # numPlayer
                    i, # playerIdx
                    False, # vdn
                    sad[i], # sad
                    hide_action[i], # hideAction
                    convention, # convention
                    act_parameterized[i], # act parameterized
                    convention_index, # conventionIndex
                    override[i], # conventionOverride
                    belief_stats, # beliefStats
                )
                if belief_stats:
                    if belief_runner is None:
                        actor.set_belief_runner_stats(None)
                    else:
                        actor.set_belief_runner_stats(belief_runner)
                actors.append(actor)
                all_actors.append(actor)
            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    if belief_runner is not None:
        belief_runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    if belief_runner is not None:
        belief_runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect, all_actors


def create_belief_runner(belief_model_path, device):
    belief_model = None

    if belief_model_path != "None":
        from belief_model import ARBeliefModel

        belief_config = utils.get_train_config(belief_model_path)

        belief_model = ARBeliefModel.load(
            belief_model_path,
            device,
            5,
            10,
            belief_config["fc_only"],
            belief_config["parameterized_belief"],
            belief_config["num_conventions"],
        )

    belief_runner = None

    if belief_model is not None:
        belief_runner = rela.BatchRunner(
                belief_model, belief_model.device, 5000, ["sample"])


    return belief_runner


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    overwrite=None,
    num_run=1,
    verbose=True,
    device="cuda:0",
    convention="None",
    override=[0, 0],
    belief_stats=False,
):
    agents = []
    sad = []
    hide_action = []
    belief_model_path="None"
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = device
    overwrite["boltzmann_act"] = False

    # Load models from weight files
    for weight_file in weight_files:
        if "rulebot" in weight_file:
            agents.append(weight_file)
            sad.append(False)
            hide_action.append(False)
            continue

        try: 
            state_dict = torch.load(weight_file)
        except:
            sys.exit(f"weight_file {weight_file} can't be loaded")

        if "fc_v.weight" in state_dict.keys():
            agent, cfg = utils.load_agent(weight_file, overwrite)
            agents.append(agent)
            sad.append(cfg["sad"] if "sad" in cfg else cfg["greedy_extra"])
            hide_action.append(bool(cfg["hide_action"]))
            if belief_stats:
                belief_model_path = cfg["belief_model"]
        else:
            agent = utils.load_supervised_agent(weight_file, "cuda:0")
            agents.append(["r2d2", agent])
            sad.append(False)
            hide_action.append(False)
        agent.train(False)

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p, games = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,  # eps
            sad,
            hide_action,
            device=device,
            convention=load_convention(convention),
            override=override,
            belief_stats=belief_stats,
            belief_model_path=belief_model_path,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print(
            "score: %.3f +/- %.3f" % (mean, sem),
            "; perfect: %.2f%%" % (100 * perfect_rate),
        )
    return mean, sem, perfect_rate, scores, games

def load_convention(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)
