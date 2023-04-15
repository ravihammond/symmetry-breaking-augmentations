import sys
import os
import argparse
import pprint
pprint = pprint.pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from easydict import EasyDict as edict
import itertools
from collections import defaultdict

from create import *
import rela
import r2d2
import utils
import csv


def similarity(args):
    colour_permutes, inverse_colour_permutes = get_colour_permutes()

    agent1 = load_agent(
            args.policy1, 
            args.sad_legacy1, 
            args.device, 
            args.shuffle_colour1, 
            colour_permutes[args.permute_index1],
            inverse_colour_permutes[args.permute_index1],
            args.name1,
    )
    agent2 = load_agent(
            args.policy2, 
            args.sad_legacy2, 
            args.device, 
            args.shuffle_colour2, 
            colour_permutes[args.permute_index2],
            inverse_colour_permutes[args.permute_index2],
            args.name2,
    )

    sp1_actors = []
    sp2_actors = []
    seed = args.seed
    _, sp1_actors = play_games(args, [agent1, agent1], [agent2, agent2], seed)
    seed += args.num_game
    _, sp2_actors = play_games(args, [agent2, agent2], [agent1, agent1], seed)
    seed += args.num_game
    xp_scores, xp_actors = play_games(args, [agent1, agent2], [agent2, agent1], seed)

    mean, std, similarity = extract_data(args, xp_scores,
            [sp1_actors, sp2_actors, xp_actors])

    if args.save:
        save(args, similarity, mean, std)

    print()


def get_colour_permutes():
    colour_permute_tuples = list(itertools.permutations([0,1,2,3,4]))
    colour_permutes = [list(x) for x in colour_permute_tuples]
    inverse_colour_permutes = []
    for permute in colour_permutes:
        inv_permute = [0,1,2,3,4]
        inv_permute = sorted(inv_permute, key=lambda x: permute[x])
        for i in range(5):
            assert(inv_permute[permute[i]] == i)
        inverse_colour_permutes.append(inv_permute)

    return colour_permutes, inverse_colour_permutes


def load_agent(policy, sad_legacy, device, shuffle_colour, 
        colour_permute, inverse_colour_permute, name):
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
        agent = utils.load_sad_model(policy, device)
        cfg = default_cfg
    else:
        agent, cfg = utils.load_agent(policy, {
            "vdn": False,
            "device": device,
            "uniform_priority": True,
        })

    if agent.boltzmann:
        boltzmann_beta = utils.generate_log_uniform(
            1 / cfg["max_t"], 1 / cfg["min_t"], cfg["num_t"]
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
    else:
        boltzmann_t = []

    if not shuffle_colour:
        colour_permute = []
        inverse_colour_permute = []

    # colour_permutes, inverse_colour_permutes = get_colour_permutes()
    # if shuffle_colour:
        # for i, permute in enumerate(colour_permutes):
            # same = True
            # for j in range(5):
                # if permute[j] != inverse_colour_permute[j]:
                    # same = False
                    # break
            # if not same:
                # continue
            # print("match found")
            # print("found inverse permute:", permute)
            # print("known inverse permute:", inverse_colour_permute)
            # print("known permute:", colour_permute)
            # print("i:", i)

    return {
        "agent": agent, 
        "cfg": cfg, 
        "boltzmann_t": boltzmann_t, 
        "sad_legacy": sad_legacy,
        "shuffle_colour": shuffle_colour,
        "colour_permute": colour_permute,
        "inverse_colour_permute": inverse_colour_permute,
        "name": name,
    }


def create_agent_str(agent):
    name = agent["name"]
    permute = ""
    if agent["shuffle_colour"]:
        permutes = ",".join(str(x) for x in agent["colour_permute"])
        permute = f"[{permutes}]"
    return f"{name}{permute}"


def play_games(args, agents, comp_agents, seed):
    if args.verbose:
        agent1_str = create_agent_str(agents[0])
        comp_agent1_str = create_agent_str(comp_agents[0])
        agent2_str = create_agent_str(agents[1])
        comp_agent2_str = create_agent_str(comp_agents[1])
        print(f"running: {agent1_str} ({comp_agent1_str})vs" + \
              f"{agent2_str} ({comp_agent2_str})")
    if args.num_game < args.num_thread:
        args.num_thread = args.num_game

    num_player = 2
    bomb = 0
    max_len = 80

    runners = []
    for i, agent in enumerate(agents):
        runners.append(rela.BatchRunner(
            agent["agent"].clone(args.device), 
            args.device, 
            1000, 
            ["act"]
        ))

    comp_runners = []
    for i, agent in enumerate(comp_agents):
        comp_runners.append(rela.BatchRunner(
            agent["agent"].clone(args.device), 
            args.device, 
            1000, 
            ["act"]
        ))

    context = rela.Context()
    threads = []

    games = create_envs(args.num_game, seed, num_player, bomb, max_len)

    assert(args.num_game % args.num_thread == 0)
    game_per_thread = args.num_game // args.num_thread
    all_actors = []

    for t_idx in range(args.num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                actor = hanalearn.R2D2Actor(
                    runners[i], # runner
                    num_player, # numPlayer
                    i, # playerIdx
                    False, # vdn
                    agents[i]["cfg"]["sad"], # sad
                    agents[i]["cfg"]["hide_action"], # hideAction
                    [], # convention
                    0, # act parameterized
                    0, # conventionIndex
                    0, # conventionOverride
                    0, # beliefStats
                    agents[i]["sad_legacy"], # sadLegacy
                    agents[i]["shuffle_colour"], #shuffleColor
                )

                actor.set_compare_runners(
                    [comp_runners[i]], 
                    [comp_agents[i]["cfg"]["sad"]],
                    [agents[i]["sad_legacy"]],
                    [comp_agents[i]["cfg"]["hide_action"]],
                    [f"comp{i}"]
                )

                actor.set_colour_permute(
                    [agents[i]["colour_permute"]],
                    [agents[i]["inverse_colour_permute"]],
                    [comp_agents[i]["shuffle_colour"]],
                    [comp_agents[i]["colour_permute"]],
                    [comp_agents[i]["inverse_colour_permute"]]
                )

                actors.append(actor)
                all_actors.append(actor)

            for i in range(num_player):
                act_partners = actors[:]
                act_partners[i] = None
                actors[i].set_partners(act_partners)

            thread_actors.append(actors)
            thread_games.append(games[g_idx])

        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True, t_idx)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    for runner in comp_runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    for runner in comp_runners:
        runner.stop()

    scores = [g.last_episode_score() for g in games]

    return scores, all_actors


def extract_data(args, scores, all_actors):
    stats = defaultdict(int)
    for actors in all_actors:
        for i, actor in enumerate(actors):
            actor_stats = defaultdict(int, actor.get_stats())
            for i in range(80):
                stats[f"turn_{i}_same"] += actor_stats[f"turn_{i}_same"]
                stats[f"turn_{i}_different"] += actor_stats[f"turn_{i}_different"]

    similarity_all = np.zeros((80, 2), dtype=int)

    for i in range(80):
        similarity_all[i][0] = stats[f"turn_{i}_same"]
        similarity_all[i][1] = stats[f"turn_{i}_different"]

    mean_sim = np.mean(similarity_all, axis=0)
    similarity = mean_sim[0] / np.sum(mean_sim)

    mean_score = np.mean(scores)
    sem_score = np.std(scores) / args.num_game

    if args.verbose:
        print(f"similarity: {similarity:.4f}")
        print(f"score: {mean_score:.4f} ± {sem_score:.4f}")

    return mean_score, sem_score, similarity


# def save_data(args, ):


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy1", type=str, required=True)
    parser.add_argument("--policy2", type=str, required=True)
    parser.add_argument("--sad_legacy1", type=int, default=0)
    parser.add_argument("--sad_legacy2", type=int, default=0)
    parser.add_argument("--shuffle_colour1", type=int, default=0)
    parser.add_argument("--shuffle_colour2", type=int, default=0)
    parser.add_argument("--permute_index1", type=int, default=0)
    parser.add_argument("--permute_index2", type=int, default=0)
    parser.add_argument("--name1", type=str, default="<none>")
    parser.add_argument("--name2", type=str, default="<none>")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--upload_gcloud", type=int, default=0)
    args = parser.parse_args()

    if args.outdir is not None:
        args.outdir = os.path.join("similarity", args.outdir)

    return args

if __name__ == "__main__":
    args = parse_args()
    similarity(args)

