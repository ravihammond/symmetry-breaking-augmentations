# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
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


def eval_convex_hull(args):
    scores = []
    perfect = 0
    for i in range(args.num_run):
        mean, all_scores, p = run_eval(args, i)
        args.seed = args.seed + args.num_game
        scores.extend(mean)
        perfect += p
        perfect_rate = float(p) / args.num_game
        print(f"score: {mean[0]:.3f}; perfect: {100 * perfect_rate}%")

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (args.num_game * args.num_run)
    print("Final")
    print(f"score: {mean:.3f} ± {sem:.3f}; perfect: {100 * perfect_rate}%")


def run_eval(args, split_index):
    all_weights = load_json_list(args.models)
    split_indexes = load_json_list(args.splits)[split_index]["train"]
    weights = [ all_weights[i] for i in split_indexes ]
    agents = load_agents(args, weights, split_indexes)
    # agents = load_agents(args, all_weights)

    mean, scores, perfect = evaluate(args, agents)

    return [mean], scores, perfect

def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


def load_agents(args, weights, split_indexes):
    agents = []

    for i, weight in enumerate(weights):
        agents.append(load_agent(
            weight,
            args.sad_legacy,
            "cuda:0",
            f"sad_{split_indexes[i] + 1}"
        ))

    return agents


def load_agent(policy, sad_legacy, device, name):
    if not os.path.exists(policy):
        print(f"Path {policy} doesn't exist.")
        sys.exit
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

    if sad_legacy:
        if "op" in policy:
            agent = utils.load_op_model(policy, device)
        else:
            agent = utils.load_sad_model(policy, device)
        cfg = default_cfg
    else:
        agent, cfg = utils.load_agent(policy, {
            "vdn": False,
            "device": device,
            "uniform_priority": True,
        })

    return {
        "agent": agent, 
        "cfg": cfg, 
        "sad_legacy": sad_legacy,
        "name": name,
    }


def evaluate(
    args,
    agents,
):
    """
    evaluate agents as long as they have a "act" function
    """
    if args.num_game < args.num_thread:
        args.num_thread = args.num_game

    devices = args.device.split(",")
    runners = []
    for i, agent in enumerate(agents):
        dev = devices[i % len(devices)]
        agent = agent["agent"]
        runner = rela.BatchRunner(agent.clone(dev), dev)
        runner.add_method("act", 5000)
        runners.append(runner)

    context = rela.Context()
    threads = []

    games = create_envs(args.num_game, args.seed, args.num_player, args.bomb, args.max_len)

    assert args.num_game % args.num_thread == 0
    game_per_thread = args.num_game // args.num_thread
    all_actors = []

    agent_idx = 0
    seed = args.seed

    for t_idx in range(args.num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            runner = runners[agent_idx]
            sad = agents[agent_idx]["cfg"]["sad"]
            hide_action = agents[agent_idx]["cfg"]["hide_action"]
            sad_legacy = agents[agent_idx]["sad_legacy"]

            for i in range(args.num_player):
                actor = hanalearn.R2D2Actor(
                    None, # runner
                    seed, # seed
                    args.num_player, # numPlayer
                    i, # playerIdx
                    False, # vdn
                    sad, # sad
                    hide_action, # hideAction
                    [], # convention
                    False, # act parameterized
                    0, # conventionIndex
                    0, # conventionOverride
                    False, # beliefStats
                    sad_legacy, # sadLegacy
                    False, #shuffleColor
                    True, #convexHull
                )

                actor.set_shadow_runners(
                    runners,
                    [x["cfg"]["sad"] for x in agents],
                    [x["sad_legacy"] for x in agents],
                    [x["cfg"]["hide_action"] for x in agents],
                    [x["name"] for x in agents]
                )

                actors.append(actor)
                all_actors.append(actor)


            for i in range(args.num_player):
                act_partners = actors[:]
                act_partners[i] = None
                actors[i].set_partners(act_partners)


            thread_actors.append(actors)
            thread_games.append(games[g_idx])
            agent_idx = (agent_idx + 1) % len(agents)
            seed += 1

        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True, t_idx)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])

    return np.mean(scores), scores, num_perfect


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
            belief_config["parameterized"],
            belief_config["num_conventions"],
        )

    belief_runner = None

    if belief_model is not None:
        belief_runner = rela.BatchRunner(
                belief_model, belief_model.device, 5000, ["sample"])


    return belief_runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, type=str, required=True)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--bomb", default=0, type=int)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--max_len", default=80, type=int)
    parser.add_argument("--num_game", default=5000, type=int)
    parser.add_argument("--num_run", default=1, type=int)
    parser.add_argument("--num_thread", default=10, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--sad_legacy", default=0, type=int)
    args = parser.parse_args()

    eval_convex_hull(args)
