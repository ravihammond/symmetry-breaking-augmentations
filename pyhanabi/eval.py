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
    partner_agents=None,
    partner_cfgs=None,
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
    runners = []
    for i, agent in enumerate(agents):
        if partner_agents is not None and i > 0:
            runners.append(None)
            break
        runners.append(rela.BatchRunner(agent.clone(device), device, 1000, ["act"]))

    partner_runners = []
    if partner_agents is not None:
        for agent in partner_agents:
            partner_runners.append(
                rela.BatchRunner(agent.clone(device), device, 1000, ["act"])
            )

    belief_runner = None
    if belief_stats:
        belief_runner = create_belief_runner(belief_model_path, device)

    context = rela.Context()
    threads = []

    games = create_envs(num_game, seed, num_player, bomb, max_len)

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []

    partner_idx = 0
    convention_index = 0

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []

            for i in range(num_player):
                runner = runners[i]
                sad_setting = sad[i]
                hide_action_setting = hide_action[i]
                act_parameterized_setting = act_parameterized[i]
                agent_str = "exps/poblf1_CR-P0/model0.pthw"

                if i > 0 and partner_agents is not None:
                    runner = partner_runners[partner_idx]
                    sad_setting = partner_cfgs[partner_idx]["sad"]
                    hide_action_setting = partner_cfgs[partner_idx]["hide_action"]
                    act_parameterized_setting = partner_cfgs[partner_idx]["parameterized"]
                
                actor = hanalearn.R2D2Actor(
                    runner, # runner
                    num_player, # numPlayer
                    i, # playerIdx
                    False, # vdn
                    sad_setting, # sad
                    hide_action_setting, # hideAction
                    convention, # convention
                    act_parameterized_setting, # act parameterized
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
            if len(partner_runners) > 0:
                partner_idx = (partner_idx + 1) % len(partner_runners)
            if len(convention) > 0:
                convention_index = (convention_index + 1) % len(convention)

        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        if runner is not None:
            runner.start()

    for runner in partner_runners:
        runner.start()

    if belief_runner is not None:
        belief_runner.start()

    context.start()
    context.join()

    for runner in runners:
        if runner is not None:
            runner.stop()

    for runner in partner_runners:
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
    partner_models_path="None",
):
    agents = []
    sad = []
    hide_action = []
    parameterized = []
    belief_model_path="None"
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = device
    overwrite["boltzmann_act"] = False

    # Load models from weight files
    for i, weight_file in enumerate(weight_files):
        if i > 0 and partner_models_path is not "None":
            agents.append(None)
            sad.append(False)
            hide_action.append(False)
            parameterized.append(False)
            break

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
            parameterized.append(cfg["parameterized"])
        else:
            agent = utils.load_supervised_agent(weight_file, "cuda:0")
            agents.append(agent)
            sad.append(False)
            hide_action.append(False)
            parameterized.append(False)
        agent.train(False)

    partner_agents, partner_cfgs = load_partner_agents(partner_models_path)

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
            convention=load_json_list(convention),
            override=override,
            belief_stats=belief_stats,
            belief_model_path=belief_model_path,
            partner_agents=partner_agents,
            partner_cfgs=partner_cfgs,
            act_parameterized=parameterized,
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

def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def load_partner_agents(partner_models):
    if partner_models is "None":
        return None, None

    partner_model_paths = load_json_list(partner_models)

    partner_agents = []
    partner_cfgs = []

    for partner_model_path in partner_model_paths:
        partner_cfg = {"sad": False, "hide_action": False}

        overwrite = {}
        overwrite["vdn"] = False
        overwrite["device"] = "cuda:0"
        overwrite["boltzmann_act"] = False
        try: 
            state_dict = torch.load(partner_model_path)
        except:
            sys.exit(f"weight_file {partner_model_path} can't be loaded")

        if "fc_v.weight" in state_dict.keys():
            partner_agent, cfg = utils.load_agent(
                    partner_model_path, overwrite)
            partner_cfg["sad"] = cfg["sad"] if "sad" in cfg else cfg["greedy_extra"]
            partner_cfg["hide_action"] = bool(cfg["hide_action"])
            partner_cfg["parameterized"] = bool(cfg["parameterized"])
            partner_cfg["weight"] = partner_model_path
        else:
            partner_agent = utils.load_supervised_agent(
                    args.partner_agent, args.act_device)
            partner_cfg["sad"] = False
            partner_cfg["hide_action"] = False
            partner_cfg["parameterized"] = False

        partner_agents.append(partner_agent)
        partner_cfgs.append(partner_cfg)

    return partner_agents, partner_cfgs
