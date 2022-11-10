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

from create import *
import rela
import sad_r2d2
import sad_utils


def evaluate(
    agents, 
    num_game, 
    seed, 
    bomb, 
    eps, 
    sad, 
    *, 
    num_thread=10, 
    hand_size=5, 
    runners=None, 
    max_len=80,
    device="cuda:0"
):
    """
    evaluate agents as long as they have a "act" function
    """
    if num_game < num_thread:
        num_thread = num_game

    assert agents is None or runners is None
    if agents is not None:
        runners = [rela.BatchRunner(agent, device, 1000, ["act"]) for agent in agents]
    num_player = len(runners)

    context = rela.Context()

    games = create_envs(num_game, seed, num_player, bomb, max_len)

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []

            for i in range(num_player):
                actors.append(hanalearn.SADActor(
                    runners[i], 
                    1,
                    i,
                ))

            thread_actors.append(actors)
            thread_games.append(games[g_idx])

        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    overwrite=None,
    num_run=1,
    verbose=True,
):
    agents = []
    sad = []
    hide_action = []
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = "cuda:0"
    overwrite["boltzmann_act"] = False

    for weight_file in weight_files:
        agent, cfg = utils.load_agent(
            weight_file,
            overwrite,
        )
        agents.append(agent)
        sad.append(cfg["sad"] if "sad" in cfg else cfg["greedy_extra"])
        hide_action.append(bool(cfg["hide_action"]))

    hand_size = cfg.get("hand_size", 5)

    assert all(s == sad[0] for s in sad)
    sad = sad[0]
    if all(h == hide_action[0] for h in hide_action):
        hide_action = hide_action[0]
        process_game = None
    else:
        hide_actions = hide_action
        process_game = lambda g: g.set_hide_actions(hide_actions)
        hide_action = False

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p, _ = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,  # eps
            sad,
            hide_action,
            process_game=process_game,
            hand_size=hand_size,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: %.2f%%" % (100 * perfect_rate))
    return mean, sem, perfect_rate, scores

