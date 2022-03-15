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

from create import *
import rela
import r2d2 
import utils

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
    convention_sender=0,
    override=[0, 0],
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
    runners = [
        rela.BatchRunner(agent, device, 1000, ["act"]) 
        if isinstance(agent, r2d2.R2D2Agent)
        else None
        for agent in agents
    ]

    # Which actor is the sender and responder of the convention
    convention_role = [1, 0]
    if convention_sender == 1:
        convention_role = [0, 1]

    context = rela.Context()
    games = create_envs(num_game, seed, num_player, bomb, max_len)
    threads = []

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                if agents[i] == "rulebot":
                    actor = hanalearn.RulebotActor(
                        i, 
                        convention, 
                        convention_role[i], 
                        override[i])
                elif agents[i] == "rulebot2":
                    actor = hanalearn.Rulebot2Actor(
                        i, 
                        convention, 
                        convention_role[i], 
                        override[i])
                else:
                    actor = hanalearn.R2D2ConventionActor(
                        runners[i], 
                        num_player, 
                        i, 
                        False, 
                        sad[i], 
                        hide_action[i],
                        convention, 
                        convention_role[i], 
                        override[i])
                    # actor = hanalearn.R2D2Actor(
                        # runners[i], num_player, i, False, sad[i], hide_action[i]
                    # )
                actors.append(actor)
                all_actors.append(actor)
            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        if isinstance(runner, rela.BatchRunner):
            runner.start()

    context.start()
    context.join()

    for runner in runners:
        if isinstance(runner, rela.BatchRunner):
            runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect, all_actors


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    overwrite=None,
    num_run=1,
    verbose=True,
    convention="None",
    convention_sender=0,
    override=[0, 0],
):
    agents = []
    sad = []
    hide_action = []
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = "cuda:0"
    overwrite["boltzmann_act"] = False

    # Load models from weight files
    for weight_file in weight_files:
        if "rulebot" in weight_file:
            agents.append(weight_file)
            sad.append(False)
            hide_action.append(False)

        try: 
            state_dict = torch.load(weight_file)
        except:
            sys.exit(f"weight_file {weight_file} can't be loaded")

        if "fc_v.weight" in state_dict.keys():
            agent, cfg = utils.load_agent(weight_file, overwrite)
            agents.append(agent)
            sad.append(cfg["sad"] if "sad" in cfg else cfg["greedy_extra"])
            hide_action.append(bool(cfg["hide_action"]))
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
            convention=load_convention(convention),
            convention_sender=convention_sender,
            override=override,
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
