# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import set_path

set_path.append_sys_path()

import rela
import hanalearn

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")


class ActGroup:
    def __init__(
        self,
        devices,
        agent,
        seed,
        num_thread,
        num_game_per_thread,
        num_player,
        explore_eps,
        boltzmann_t,
        method,
        sad,
        shuffle_color,
        hide_action,
        trinary,
        replay_buffer,
        multi_step,
        max_len,
        gamma,
        off_belief,
        belief_model,
        actor_type,
        convention,
    ):
        self.devices = devices.split(",")
        self.method = method
        self.seed = seed
        self.num_thread = num_thread
        self.num_player = num_player
        self.num_game_per_thread = num_game_per_thread
        self.explore_eps = explore_eps
        self.boltzmann_t = boltzmann_t
        self.sad = sad
        self.shuffle_color = shuffle_color
        self.hide_action = hide_action
        self.trinary = trinary
        self.replay_buffer = replay_buffer
        self.multi_step = multi_step
        self.max_len = max_len
        self.gamma = gamma

        self.model_runners = []
        for dev in self.devices:
            runner = rela.BatchRunner(agent.clone(dev), dev)
            runner.add_method("act", 5000)
            runner.add_method("compute_priority", 100)
            if off_belief:
                runner.add_method("compute_target", 5000)
            self.model_runners.append(runner)
        self.num_runners = len(self.model_runners)

        self.off_belief = off_belief
        self.belief_model = belief_model
        self.belief_runner = None
        if belief_model is not None:
            self.belief_runner = []
            for bm in belief_model:
                print("add belief model to: ", bm.device)
                self.belief_runner.append(
                    rela.BatchRunner(bm, bm.device, 5000, ["sample"])
                )
        self.convention = convention

        Actor = None
        if actor_type == "r2d2":
            Actor = hanalearn.R2D2Actor
        elif actor_type == "r2d2_convention":
            Actor = hanalearn.R2D2ConventionActor
        assert Actor is not None

        self.create_r2d2_actors(hanalearn.R2D2ConventionActor)
        print("ActGroup created")

    def create_r2d2_actors(self, Actor):
        actors = []
        if self.method == "vdn":
            for i in range(self.num_thread):
                thread_actors = []
                for j in range(self.num_game_per_thread):
                    actor = hanalearn.R2D2Actor(
                        self.model_runners[i % self.num_runners],
                        self.seed,
                        self.num_player,
                        0,
                        self.explore_eps,
                        self.boltzmann_t,
                        True,
                        self.sad,
                        self.shuffle_color,
                        self.hide_action,
                        self.trinary,
                        self.replay_buffer,
                        self.multi_step,
                        self.max_len,
                        self.gamma,
                        self.convention,
                        0,
                        0,
                    )
                    self.seed += 1
                    thread_actors.append([actor])
                actors.append(thread_actors)
        elif self.method == "iql":
            for i in range(self.num_thread):
                thread_actors = []
                for j in range(self.num_game_per_thread):
                    game_actors = []
                    for k in range(self.num_player):
                        actor = hanalearn.R2D2Actor(
                            self.model_runners[i % self.num_runners],
                            self.seed,
                            self.num_player,
                            k,
                            self.explore_eps,
                            self.boltzmann_t,
                            False,
                            self.sad,
                            self.shuffle_color,
                            self.hide_action,
                            self.trinary,
                            self.replay_buffer,
                            self.multi_step,
                            self.max_len,
                            self.gamma,
                            self.convention,
                            0,
                            0,
                        )
                        if self.off_belief:
                            if self.belief_runner is None:
                                actor.set_belief_runner(None)
                            else:
                                actor.set_belief_runner(
                                    self.belief_runner[i % len(self.belief_runner)]
                                )
                        self.seed += 1
                        game_actors.append(actor)
                    for k in range(self.num_player):
                        partners = game_actors[:]
                        partners[k] = None
                        game_actors[k].set_partners(partners)
                    thread_actors.append(game_actors)
                actors.append(thread_actors)
        self.actors = actors

    def start(self):
        for runner in self.model_runners:
            runner.start()

        if self.belief_runner is not None:
            for runner in self.belief_runner:
                runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)
