import sys
import os
import argparse
import pprint
pprint = pprint.pprint
import numpy as np
import matplotlib.pyplot as plt
import torch

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

def calculate_policy_differences(
    act_policy1,
    act_policy2,
    comp_policies,
    num_game,
    num_thread,
    seed,
    batch_size,
    act_sad_legacy,
    comp_sad_legacy,
    device,
    comp_names,
    output_dir,
    rand_policy,
    verbose,
    compare_as_base,
):
    print("\nact policy")
    act_agent = load_agent(act_policy, act_sad_legacy, device)
    print("comp policies")
    comp_agents = load_agents(comp_policies, comp_sad_legacy, device)

    replay_buffer = generate_replay_data(
        act_agent,
        comp_agents,
        num_game,
        seed,
        0,
        num_thread=num_thread,
        comp_names=comp_names
    )

    comp_base_sad_legacy = None
    if compare_as_base != "None":
        print(f"Using {compare_as_base} as comparision base.")
        comp_base_sad_legacy = comp_sad_legacy[comp_names.index(compare_as_base)]
        comp_names.remove(compare_as_base)

    data = extract_data(replay_buffer, batch_size, "cpu", comp_names,
            act_sad_legacy, comp_sad_legacy, rand_policy, 
            compare_as_base, comp_base_sad_legacy, verbose)

    if output_dir is not None:
        save_data(data, act_policy, comp_policies, comp_names, output_dir, rand_policy)


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
    act_agent,
    comp_agents,
    num_game,
    seed,
    bomb,
    *,
    num_thread=10,
    max_len=80,
    device="cuda:0",
    num_player=2,
    comp_names=[],
):
    agent, cfgs, boltzmann_t, sad_legacy = act_agent

    if num_game < num_thread:
        num_thread = num_game

    act_runner = rela.BatchRunner(agent.clone(device), device)
    act_runner.add_method("act", 5000)
    act_runner.add_method("compute_priority", 100)

    comp_runners = []
    comp_sad = []
    comp_sad_legacy = []
    comp_hide_action = []

    for comp_agent_tuple in comp_agents:
        comp_agent, comp_cfgs, _, comp_sad_legacy_temp = comp_agent_tuple
        comp_runner = rela.BatchRunner(comp_agent.clone(device), device)
        comp_runner.add_method("act", 5000)
        comp_runners.append(comp_runner)
        comp_sad.append(comp_cfgs["sad"])
        comp_sad_legacy.append(comp_sad_legacy_temp)
        comp_hide_action.append(comp_cfgs["hide_action"])

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

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                actor = hanalearn.R2D2Actor(
                    act_runner, # runner
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
                    sad_legacy, # sadLegacy
                    False, # beliefSadLegacy
                    False, # colorShuffleSync
                )

                if len(comp_runners) > 0:
                    actor.set_compare_runners(
                        comp_runners, 
                        comp_sad,
                        comp_sad_legacy,
                        comp_hide_action,
                        comp_names)

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

    act_runner.start()

    for runner in comp_runners:
        runner.start()

    context.start()
    context.join()

    act_runner.stop()

    for runner in comp_runners:
        runner.stop()

    return replay_buffer


def extract_data(
        replay_buffer, 
        batch_size, 
        device, 
        comp_names, 
        act_sad_legacy, 
        comp_sad_legacy, 
        rand_policy, 
        compare_as_base,
        comp_base_sad_legacy,
        verbose):
    assert(replay_buffer.size() % batch_size == 0)
    num_batches = (int)(replay_buffer.size() / batch_size)

    for batch_index in range(num_batches):
        range_start = batch_index * batch_size
        range_end = batch_index * batch_size + batch_size
        sample_id_list = [*range(range_start, range_end, 1)]

        batch, _ = replay_buffer.sample_from_list(
                batch_size, device, sample_id_list)

        similarities = get_similarities(batch, comp_names, act_sad_legacy, 
                comp_sad_legacy, rand_policy, compare_as_base, 
                comp_base_sad_legacy, verbose)

    return similarities


def get_similarities(batch, comp_names, act_sad_legacy, comp_sad_legacy, 
        rand_policy, compare_as_base, comp_base_sad_legacy, verbose):
    seq_len = np.array(batch.seq_len)

    actions = np.array(batch.action["a"])
    if not act_sad_legacy:
        actions = np.expand_dims(actions, axis=2)
    if compare_as_base != "None":
        action_key = compare_as_base + ":a"
        actions = np.array(batch.action[action_key])
        if not comp_base_sad_legacy:
            actions = np.expand_dims(actions, axis=2)
    actions = clean_actions(actions, seq_len)

    comp_actions = {}
    for i, name in enumerate(comp_names):
        action_key = name + ":a"
        comp_action = np.array(batch.action[action_key])
        if not comp_sad_legacy[i]:
            comp_action = np.expand_dims(comp_action, axis=2)
        comp_action = clean_actions(comp_action, seq_len)

        comp_actions[action_key] = comp_action

    if rand_policy:
        create_random_actions(batch, actions, comp_actions, comp_names)

    if verbose:
        print_games(actions, comp_actions, comp_names)

    similarities = {}
    for name in comp_names:
        action_key = name + ":a"

        # Sum number of same actions
        action_diff = np.array(actions == comp_actions[action_key], dtype=int)
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

        similarities[action_key] = similarity.squeeze(axis=1)

    return similarities


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


def save_data(data, act_policy, comp_policies, comp_names, output_dir, rand_policy):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    act_policy_no_ext = os.path.splitext(act_policy)[0]
    base_actor = os.path.basename(act_policy_no_ext)
    # base_actor = os.path.basename(os.path.dirname(act_policy))

    for i, comp_name in enumerate(comp_names):
        if comp_name == "rand":
            comp_full_name = comp_name
        else:
            comp_full_name = os.path.basename(os.path.dirname(comp_policies[i]))
        filename = comp_full_name + "_vs_" + base_actor + ".csv"

        save_dir = os.path.join(output_dir, comp_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, filename)
        print("saving:", save_path)
        data_key = comp_name + ":a"
        np.savetxt(save_path, data[data_key], delimiter=",",fmt='%1.4f')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act_policy1", type=str, required=True)
    parser.add_argument("--act_policy2", type=str, required=True)
    parser.add_argument("--comp_policies", type=str, default="None")
    parser.add_argument("--act_sad_legacy", type=str, default="0,0")
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
        args.outdir = os.path.join("similarity_data", args.out_dir)

    return args

if __name__ == "__main__":
    args = parse_args()

    calculate_policy_differences(
        args.act_policy1,
        args.act_policy2,
        args.comp_policies,
        args.num_game,
        args.num_thread,
        args.seed,
        args.batch_size,
        args.act_sad_legacy,
        args.comp_sad_legacy,
        args.device,
        args.comp_names,
        args.outdir,
        args.rand_policy,
        args.verbose,
        args.compare_as_base,
    )
