# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import set_path
set_path.append_sys_path()

import os
import pprint
import time
import copy

import numpy as np
import torch
import rela
import hanalearn

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")


def create_envs(
    num_env,
    seed,
    num_player,
    hand_size,
    bomb,
    explore_eps,
    max_len,
    sad,
    shuffle_obs,
    shuffle_color,
):
    games = []
    for game_idx in range(num_env):
        params = {
            "players": str(num_player),
            "hand_size": str(hand_size),
            "seed": str(seed + game_idx),
            "bomb": str(bomb),
        }
        game = hanalearn.HanabiEnv(
            params,
            explore_eps,
            max_len,
            sad,
            shuffle_obs,
            shuffle_color,
            False,
        )
        games.append(game)
    return games


def create_threads(
    num_thread,
    num_game_per_thread,
    actors,
    games,
):
    context = rela.Context()
    threads = []
    for thread_idx in range(num_thread):
        env = hanalearn.HanabiVecEnv()
        for game_idx in range(num_game_per_thread):
            env.append(games[thread_idx * num_game_per_thread + game_idx])
        thread = hanalearn.HanabiThreadLoop(actors[thread_idx], env, False)
        threads.append(thread)
        context.push_env_thread(thread)
    print(
        "Finished creating %d threads with %d games and %d actors"
        % (len(threads), len(games), len(actors))
    )
    return context, threads


class ActGroup:
    def __init__(
        self,
        method,
        devices,
        agent,
        num_thread,
        num_game_per_thread,
        multi_step,
        gamma,
        eta,
        max_len,
        num_player,
        replay_buffer,
        shuffle_color,
    ):
        self.devices = devices.split(",")

        self.model_runners = []
        for dev in self.devices:
            runner = rela.BatchRunner(
                agent.clone(dev), dev, 100, ["act", "compute_priority"]
            )
            self.model_runners.append(runner)

        self.num_runners = len(self.model_runners)

        actors = []
        if method == "vdn":
            for i in range(num_thread):
                actor = hanalearn.SADActor(
                    self.model_runners[i % self.num_runners],
                    multi_step,
                    num_game_per_thread,
                    gamma,
                    eta,
                    max_len,
                    num_player,
                    replay_buffer,
                )
                actors.append(actor)
        elif method == "iql":
            for i in range(num_thread):
                thread_actors = []
                for j in range(num_game_per_thread):
                    game_actors = []
                    for k in range(num_player):
                        # actor = hanalearn.SADActor(
                            # self.model_runners[i % self.num_runners],
                            # num_game_per_thread,
                            # eta,
                            # 1,
                            # k,
                            # shuffle_color,
                            # replay_buffer,
                        # )
                        actor = hanalearn.R2D2Actor(
                            self.model_runners[i % self.num_runners], # runner
                            0, # seed
                            num_player, # numPlayer
                            k, # playerIdx
                            [0], # epsList
                            [], # tempList
                            False, # vdn
                            True, # sad
                            shuffle_color, # shuffleColor
                            False, # hideAction 
                            True, # trinary
                            replay_buffer, # replayBuffer
                            multi_step, # multiStep
                            max_len, # seqLen
                            gamma, # gamma
                            [], # convention
                            False, # actParameterized
                            0, # conventionIdx
                            0, # conventionOverride
                            False, # conventionFectitiousOverride
                            True, # useExperience
                            False, # beliefStats
                            True, # sadLegacy
                        )
                        game_actors.append(actor)

                    thread_actors.append(game_actors)

                actors.append(thread_actors)

        self.actors = actors
        print("ActGroup created")

    def start(self):
        for runner in self.model_runners:
            runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)

