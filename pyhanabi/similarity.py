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
import pathlib
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from functools import partial

from create import *
import rela
import r2d2
import utils
import csv

from google_cloud_handler import GoogleCloudUploader


def similarity(args):
    print(f"{args.name1} vs {args.name2}")
    jobs = create_similarity_jobs(args)

    agents, runners = load_runners(args)

    for runner in runners:
        runner.start()

    run_job_fixed=partial(
        run_job, 
        agents=agents, 
        runners=runners,
    )

    for job in jobs:
        run_job(job, agents, runners)

    for runner in runners:
        runner.stop()


def create_similarity_jobs(args):
    jobs = []
    shuffle_indexes = parse_numbers(args.shuffle_index)

    job = edict()

    for shuffle_index in shuffle_indexes:
        job2 = copy.copy(job)
        job2.permute_index = shuffle_index
        jobs.append(job2)

    return jobs


def parse_numbers(numbers):
    if '-' in numbers:
        range = [int(x) for x in numbers.split('-')]
        assert(len(range) == 2)
        numbers_list = list(np.arange(*range))
    else:
        numbers_list = [int(x) for x in numbers.split(',')]

    return numbers_list


def load_runners(args):
    colour_permutes, inverse_colour_permutes = get_colour_permutes()

    agent1 = load_agent(
            args.policy1, 
            args.sad_legacy1, 
            args.device, 
            True, 
            args.name1,
    )

    agent2 = load_agent(
            args.policy2, 
            args.sad_legacy2, 
            args.device, 
            False,
            args.name2,
    )

    agents = [agent1, agent2]

    runners = []

    for i, agent in enumerate(agents):
        runners.append(rela.BatchRunner(
            agent["agent"].clone(args.device), 
            args.device, 
            1000, 
            ["act"]
        ))

    return agents, runners


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


def load_agent(policy, sad_legacy, device, shuffle_colour, name):
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

    return {
        "agent": agent, 
        "cfg": cfg, 
        "boltzmann_t": boltzmann_t, 
        "sad_legacy": sad_legacy,
        "shuffle_colour": shuffle_colour,
        "name": name,
    }


def run_job(job, agents, runners):
    all_colour_permutes, all_inverse_colour_permutes = get_colour_permutes()
    colour_permutes = [all_colour_permutes[job.permute_index], []]
    inverse_colour_permutes = [all_inverse_colour_permutes[job.permute_index], []]

    sp1_actors = []
    sp2_actors = []
    seed = args.seed
    sp1_scores, sp1_actors = play_games(args, agents, runners, colour_permutes, 
            inverse_colour_permutes, [0, 0], [1, 1], seed)
    seed += args.num_game
    _, sp2_actors = play_games(args, agents, runners, colour_permutes, 
            inverse_colour_permutes, [1, 1], [0, 0], seed)
    seed += args.num_game
    xp_scores, xp_actors = play_games(args, agents, runners, colour_permutes, 
            inverse_colour_permutes, [0, 1], [1, 0], seed)

    similarity, mean_score, sem_score = extract_data(args, xp_scores,
            [sp1_actors, sp2_actors, xp_actors])

    if args.save:
        save_and_upload(args, similarity, mean_score, 
                sem_score, job.permute_index)


def create_agent_str(agent, colour_permute):
    name = agent["name"]
    permute = ""
    if agent["shuffle_colour"]:
        permutes = ",".join(str(x) for x in colour_permute)
        permute = f"[{permutes}]"
    return f"{name}{permute}"


def play_games(args, agents, runners, colour_permutes, 
        inverse_colour_permutes, act_index, comp_index, seed):
    actor_agents = [agents[i] for i in act_index]
    actor_runners = [runners[i] for i in act_index]
    actor_colour_permutes = [colour_permutes[i] for i in act_index]
    actor_inverse_colour_permutes = [inverse_colour_permutes[i] for i in act_index]

    comp_agents = [agents[i] for i in comp_index]
    comp_runners = [runners[i] for i in comp_index]
    comp_colour_permutes = [colour_permutes[i] for i in comp_index]
    comp_inverse_colour_permutes = [inverse_colour_permutes[i] for i in comp_index]

    if args.verbose:
        agent1_str = create_agent_str(
                actor_agents[0], colour_permutes[act_index[0]])
        comp_agent1_str = create_agent_str(
                comp_agents[0], colour_permutes[comp_index[0]])
        agent2_str = create_agent_str(
                actor_agents[1], colour_permutes[act_index[1]])
        comp_agent2_str = create_agent_str(
                comp_agents[1], colour_permutes[comp_index[1]])
        print(f"running: {agent1_str} ({comp_agent1_str})  vs  " + \
              f"{agent2_str} ({comp_agent2_str})")

    if args.num_game < args.num_thread:
        args.num_thread = args.num_game

    num_player = 2
    bomb = 0
    max_len = 80

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
                    actor_runners[i], # runner
                    num_player, # numPlayer
                    i, # playerIdx
                    False, # vdn
                    actor_agents[i]["cfg"]["sad"], # sad
                    actor_agents[i]["cfg"]["hide_action"], # hideAction
                    [], # convention
                    0, # act parameterized
                    0, # conventionIndex
                    0, # conventionOverride
                    0, # beliefStats
                    actor_agents[i]["sad_legacy"], # sadLegacy
                    actor_agents[i]["shuffle_colour"], #shuffleColor
                )

                actor.set_compare_runners(
                    [comp_runners[i]], 
                    [comp_agents[i]["cfg"]["sad"]],
                    [comp_agents[i]["sad_legacy"]],
                    [comp_agents[i]["cfg"]["hide_action"]],
                    [comp_agents[i]["name"]],
                )

                actor.set_colour_permute(
                    [actor_colour_permutes[i]],
                    [actor_inverse_colour_permutes[i]],
                    [comp_agents[i]["shuffle_colour"]],
                    [comp_colour_permutes[i]],
                    [comp_colour_permutes[i]]
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

    context.start()
    context.join()

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

    return similarity_all, mean_score, sem_score


def save_and_upload(args, similarity, mean_score, sem_score, permute_index):
    if not args.save:
        return

    model1 = args.name1.split("_")[0]
    model2 = args.name2.split("_")[0]
    pair_name = f"{args.name1}_vs_{args.name2}"
    outdir = os.path.join(
        args.outdir, 
        f"{model1}_vs_{model2}",
        pair_name
    )


    similarity_dir = os.path.join(outdir, "similarity")
    scores_dir = os.path.join(outdir, "scores")
    if not os.path.exists(similarity_dir):
        os.makedirs(similarity_dir)
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)


    filename = f"{pair_name}_permute_{permute_index}.csv"
    similarity_save_path = os.path.join(similarity_dir, filename)
    scores_save_path = os.path.join(scores_dir, filename)

    scores = np.array([mean_score, sem_score])

    print("saving similarity:", similarity_save_path)
    np.savetxt(similarity_save_path, similarity, delimiter=",",fmt='%i')

    print("saving scores:", scores_save_path)
    np.savetxt(scores_save_path, scores, delimiter=",",fmt='%f')

    if not args.upload_gcloud:
        return

    print(args.gcloud_dir)
    similarity_path_obj = pathlib.Path(similarity_save_path)
    gc_similarity_path = os.path.join(args.gcloud_dir, *similarity_path_obj.parts[1:])
    scores_path_obj = pathlib.Path(scores_save_path)
    gc_scores_path = os.path.join(args.gcloud_dir, *scores_path_obj.parts[1:])

    gc_handler = GoogleCloudUploader("aiml-reid-research", "Ravi")

    print("uploading:", gc_similarity_path)
    gc_handler.assert_file_doesnt_exist(gc_similarity_path)
    gc_handler.upload(similarity_save_path, gc_similarity_path)

    print("uploading:", gc_scores_path)
    gc_handler.assert_file_doesnt_exist(gc_scores_path)
    gc_handler.upload(scores_save_path, gc_scores_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy1", type=str, required=True)
    parser.add_argument("--policy2", type=str, required=True)
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--sad_legacy1", type=int, default=0)
    parser.add_argument("--sad_legacy2", type=int, default=0)
    parser.add_argument("--shuffle_index", type=str, default="0")
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
    parser.add_argument("--gcloud_dir", type=str, default="hanabi-similarity")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    similarity(args)

