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

from create import *
import rela
import r2d2
import utils
import csv

ACTION_TO_STRING = {
    0: "(D0)",
    1: "(D1)",
    2: "(D2)",
    3: "(D3)",
    4: "(D4)",
    5: "(P0)",
    6: "(P1)",
    7: "(P2)",
    8: "(P3)",
    9: "(P4)",
    10: "(CR)",
    11: "(CY)",
    12: "(CG)",
    13: "(CW)",
    14: "(CB)",
    15: "(R1)",
    16: "(R2)",
    17: "(R3)",
    18: "(R4)",
    19: "(R5)",
    20: "----",
}

def run_policy_evaluation(args):
    act_policies = [args.act_policy1, args.act_policy2]
    act_agents = load_agents(act_policies, args.act_sad_legacy, args.device)
    if len(args.comp_policies) > 0:
        print("comp policies")
    comp_agents = load_agents(args.comp_policies, args.comp_sad_legacy, args.device)

    replay_buffer = generate_replay_data(args, act_agents, comp_agents)
    data = extract_data(args, replay_buffer)

    if args.outdir is not None:
        save_all_data(args, data)


def load_agents(policies, sad_legacy, device):
    agents = []
    for i, policy in enumerate(policies):
        agents.append(load_agent(policy, sad_legacy[i], device))
    return agents


def load_agent(policy, sad_legacy, device):
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

    return (agent, cfg, boltzmann_t, sad_legacy)


def generate_replay_data(
    args,
    act_agents,
    comp_agents
):
    seed = args.seed
    num_player = 2
    num_thread = args.num_thread
    if args.num_game < num_thread:
        num_thread = args.num_game

    act_runners = []
    for agent, _, _, _ in act_agents:
        act_runner = rela.BatchRunner(agent.clone(args.device), args.device)
        act_runner.add_method("act", 5000)
        act_runner.add_method("compute_priority", 100)
        act_runners.append(act_runner)

    comp_runners = []
    comp_sad = []
    comp_sad_legacy = []
    comp_hide_action = []

    for comp_agent_tuple in comp_agents:
        comp_agent, comp_cfgs, _, comp_sad_legacy_temp = comp_agent_tuple
        comp_runner = rela.BatchRunner(comp_agent.clone(args.device), args.device)
        comp_runner.add_method("act", 5000)
        comp_runners.append(comp_runner)
        comp_sad.append(comp_cfgs["sad"])
        comp_sad_legacy.append(comp_sad_legacy_temp)
        comp_hide_action.append(comp_cfgs["hide_action"])

    context = rela.Context()
    threads = []

    games = create_envs(
        args.num_game,
        seed,
        2,
        0, 
        80
    )

    replay_buffer_size = args.num_game * 2

    replay_buffer = rela.RNNPrioritizedReplay(
        replay_buffer_size,
        seed,
        1.0,  # priority exponent
        0.0,  # priority weight
        3, #prefetch
    )

    assert args.num_game % num_thread == 0
    game_per_thread = args.num_game // num_thread
    all_actors = []

    partner_idx = 0

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                cfgs = act_agents[i][1]
                actor = hanalearn.R2D2Actor(
                    act_runners[i], # runner
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
                    [], # convention
                    cfgs["parameterized"], # actParameterized
                    0, # conventionIdx
                    0, # conventionOverride
                    False, # fictitiousOverride
                    True, # useExperience
                    False, # beliefStats
                    act_agents[i][3], # sadLegacy
                    False, # beliefSadLegacy
                    False, # colorShuffleSync
                )

                if i == 0 and len(comp_runners) > 0:
                    actor.set_compare_runners(
                        comp_runners, 
                        comp_sad,
                        comp_sad_legacy,
                        comp_hide_action,
                        args.comp_names)

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

    for runner in act_runners:
        runner.start()

    for runner in comp_runners:
        runner.start()

    context.start()
    context.join()

    for runner in act_runners:
        runner.stop()

    for runner in comp_runners:
        runner.stop()

    return replay_buffer


def extract_data(args, replay_buffer):
    assert(replay_buffer.size() % args.batch_size == 0)
    num_batches = (int)(replay_buffer.size() / args.batch_size)

    for batch_index in range(num_batches):
        range_start = batch_index * args.batch_size
        range_end = batch_index * args.batch_size + args.batch_size
        sample_id_list = [*range(range_start, range_end, 1)]

        batch, _ = replay_buffer.sample_from_list_split(
                args.batch_size, "cpu", sample_id_list)
        
        # batch = replay_buffer.sample_from_list(
                # args.batch_size, "cpu", sample_id_list)

        similarities = get_similarities(args, batch)

    return similarities


def get_similarities(args, batch):
    comp_base_sad_legacy = None

    seq_len = np.array(batch.seq_len)

    actions = np.array(batch.action["a"])

    if not args.act_sad_legacy[0]:
        actions = np.expand_dims(actions, axis=2)

    actions = clean_actions(actions, seq_len)

    comp_actions = {}
    for i, name in enumerate(args.comp_names):
        action_key = name + ":a"

        comp_action = np.array(batch.action[action_key])
        if not args.comp_sad_legacy[i]:
            comp_action = np.expand_dims(comp_action, axis=2)
        comp_action = clean_actions(comp_action, seq_len)

        comp_actions[action_key] = comp_action

    if args.rand_policy:
        create_random_actions(batch, actions, comp_actions, args.comp_names)

    if args.verbose:
        print_games(actions, comp_actions, args.comp_names)

    similarities = {}

    if args.similarity_across_all:
        if len(args.comp_names) == 0:
            return similarities

        all_diffs = []
        for name in args.comp_names:
            action_key = name + ":a"
            # Sum number of same actions
            single_diff = np.array(actions == comp_actions[action_key], dtype=int)
            all_diffs.append(single_diff)

        action_diff = all_diffs[0]
        for diff in all_diffs:
            action_diff = action_diff | diff

        similarity = calculate_similarity(actions, action_diff)
        similarities["obl:a"] = similarity.squeeze(axis=1)
    else:
        for name in args.comp_names:
            action_key = name + ":a"
            # Sum number of same actions
            action_diff = np.array(actions == comp_actions[action_key], dtype=int)

            similarity = calculate_similarity(action_diff)
            similarities[action_key] = similarity.squeeze(axis=1)

    return similarities

def calculate_similarity(actions, action_diff):
    invalid = np.array(actions == 20, dtype=int)
    diff_without_invalid = np.subtract(action_diff, invalid)
    summed = diff_without_invalid.sum(axis=1, keepdims=True)

    # Get totals
    invalid_summed = np.sum(invalid, axis=1, keepdims=True)
    totals = np.full(summed.shape, actions.shape[1])
    totals_no_invalid = np.subtract(totals, invalid_summed)

    # Calculate similarity
    similarity = np.divide(summed, totals_no_invalid,
                        out=np.zeros(summed.shape), 
                        where=totals_no_invalid!=0)

    return similarity


def clean_actions(actions, seq_len):
    seq_list = np.arange(actions.shape[0])
    mask = np.invert(seq_list < seq_len[..., None])
    mask_t = np.transpose(mask)
    seq_mask = np.expand_dims(mask_t, axis=2)
    actions[seq_mask] = 20 
    return actions


def create_random_actions(batch, actions, comp_actions, comp_names):
    comp_names.append("rand")
    seq_len = np.array(batch.seq_len)

    legal_moves = batch.action["legal_moves"]

    # Add no-op legal moves to all moves after game has finished
    seq_list = np.arange(legal_moves.shape[0])
    mask = np.invert(seq_list < seq_len[..., None])
    mask_t = np.transpose(mask)
    seq_mask = np.expand_dims(mask_t, axis=2)
    seq_mask = np.repeat(seq_mask, legal_moves.shape[2], axis=2).astype(float)
    legal_moves_shape = list(legal_moves.shape)
    legal_moves_shape[len(legal_moves_shape) - 1] -= 1
    zeros_mask = np.zeros(legal_moves_shape)
    seq_mask[:,:,0:legal_moves.shape[2] - 1] = zeros_mask
    legal_moves = legal_moves + seq_mask

    legal_moves_shape = list(legal_moves.shape)
    legal_moves_shape[0] = 0
    legal_moves_shape[len(legal_moves_shape) - 1] = 1

    rand_actions = np.empty((legal_moves_shape))

    for time_step in range(legal_moves.shape[0]):
        legal_moves_slice = legal_moves[time_step]
        rand_action = torch.multinomial(legal_moves[time_step], num_samples=1)
        rand_action = np.expand_dims(rand_action.numpy(), axis=0)
        rand_actions = np.vstack((rand_actions, rand_action))

    comp_actions["rand:a"] = rand_actions.astype(int)


def print_games(actions, comp_actions, comp_names):
    print()
    for game in range(actions.shape[1]):
        print("base", end="")
        for comp_name in comp_names:
            print(f"\t{comp_name}", end="")
        print("\t\t", end="")
    print()

    for time in range(actions.shape[0]):
        for game in range(actions.shape[1]):
            action = actions[time,game,0]
            action_str = ACTION_TO_STRING[action]
            print(f"{action} {action_str}", end="")
            for comp_name in comp_names:
                action = comp_actions[comp_name + ":a"][time,game,0]
                action_str = ACTION_TO_STRING[action]
                print(f"\t{action} {action_str}", end="")
            print("\t\t", end="")
        print()


def save_all_data(args, data):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    base_actor = os.path.basename(
            os.path.dirname(args.act_policy1))

    if "sad" in args.base_name:
        comp_policy_no_ext = os.path.splitext(args.act_policy1)[0]
        base_actor = os.path.basename(comp_policy_no_ext)

    comp_policy_no_ext = os.path.splitext(args.act_policy2)[0]
    partner_actor = os.path.basename(comp_policy_no_ext)

    filename = base_actor + "_vs_" + partner_actor + ".csv"

    if args.similarity_across_all:
        data_key = "obl:a"
        save_sata(args, data, data_key, filename)
    else: 
        for comp_name in args.comp_names:
            data_key = comp_name + ":a"
            save_sata(args, data, data_key, filename)

def save_sata(args, data, data_key, filename):
    save_dir = os.path.join(args.outdir, args.base_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, filename)
    print("saving:", save_path)
    np.savetxt(save_path, data[data_key], delimiter=",",fmt='%1.4f')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act_policy1", type=str, required=True)
    parser.add_argument("--act_policy2", type=str, required=True)
    parser.add_argument("--comp_policies", type=str, default="None")
    parser.add_argument("--act_sad_legacy", type=str, default="0,0")
    parser.add_argument("--base_name", type=str, default=None)
    parser.add_argument("--comp_sad_legacy", type=str, default="None")
    parser.add_argument("--rand_policy", type=int, default=0)
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--comp_names", type=str, default="None")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--compare_as_base", type=str, default="None")
    parser.add_argument("--similarity_across_all", type=int, default=0)
    args = parser.parse_args()

    args.act_sad_legacy = [int(x) for x in args.act_sad_legacy.split(",")]
    assert(len(args.act_sad_legacy) <= 2)
    if (len(args.act_sad_legacy) == 1):
        args.act_sad_legacy *= 2
    
    if args.comp_policies == "None":
        assert(args.comp_sad_legacy == "None")
        args.comp_policies = []
        args.comp_sad_legacy = []
    else:
        assert(args.comp_sad_legacy != "None")
        args.comp_policies = args.comp_policies.split(",")
        args.comp_sad_legacy = [int(x) for x in args.comp_sad_legacy.split(",")]
        assert(len(args.comp_sad_legacy) <= len(args.comp_policies))
        if (len(args.comp_sad_legacy) == 1):
            args.comp_sad_legacy *= len(args.comp_policies)

    if args.comp_names == "None":
        args.comp_names = []
    else:
        args.comp_names = args.comp_names.split(",")
        assert(len(args.comp_names) == len(args.comp_policies))

    if args.batch_size is None:
        args.batch_size = args.num_game * 2

    if args.outdir is not None:
        args.outdir = os.path.join("similarity_data", args.outdir)

    return args

if __name__ == "__main__":
    args = parse_args()
    run_policy_evaluation(args)

