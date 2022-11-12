# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pprint
pprint = pprint.pprint

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
        convention,
        act_parameterized,
        convention_act_override,
        convention_fict_act_override,
        partner_agents,
        partner_cfgs,
        static_partner,
        use_experience,
        belief_stats,
        sad_legacy,
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

        self.partner_runners = []
        if static_partner and partner_agents is not None:
            for dev in self.devices:
                runners = []
                for agent in partner_agents:
                    runners.append(
                            rela.BatchRunner(agent.clone(dev), dev, 5000, ["act"]))
                self.partner_runners.append(runners)

        self.off_belief = off_belief
        self.belief_model = belief_model
        self.belief_runner = None
        if belief_model is not None:
            self.belief_runner = []
            for bm in belief_model:
                print("add belief model to: ", bm.device)
                self.belief_runner.append(
                    rela.BatchRunner(bm, bm.device, 5000, ["sample"]))

        self.convention = convention
        self.act_parameterized = act_parameterized
        self.convention_act_override = convention_act_override
        self.convention_fict_act_override = convention_fict_act_override
        self.partner_cfgs = partner_cfgs
        self.static_partner = static_partner
        self.use_experience = use_experience
        self.belief_stats = belief_stats
        self.sad_legacy = sad_legacy

        if self.sad_legacy:
            self.trinary = True

        self.create_r2d2_actors()

    def create_r2d2_actors(self):
        convention_index_count = 0

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
                        0, # act paramaterized
                        0, # convention index
                        0, # convention act override
                        0, # convention fict act override
                        1, # convention use experience
                    )
                    self.seed += 1
                    thread_actors.append([actor])
                actors.append(thread_actors)
        elif self.method == "iql":

            partner_idx = -1
            convention_index = -1
            for i in range(self.num_thread):
                thread_actors = []
                for j in range(self.num_game_per_thread):
                    game_actors = []

                    if len(self.convention) > 0:
                        convention_index = (convention_index + 1) % len(self.convention)

                    if len(self.partner_runners) > 0:
                        partner_idx = (partner_idx + 1) % len(self.partner_runners)

                    for k in range(self.num_player):
                        runner = self.model_runners[i % self.num_runners]
                        sad = self.sad
                        hide_action = self.hide_action
                        weight = "cosca"
                        if k > 0 and self.static_partner:
                            runner = self.partner_runners[i % self.num_runners][partner_idx]
                            sad = self.partner_cfgs[partner_idx]["sad"]
                            hide_action = self.partner_cfgs[partner_idx]["hide_action"]
                            weight = self.partner_cfgs[partner_idx]["weight"]

                        actor = hanalearn.R2D2Actor(
                            runner,
                            self.seed,
                            self.num_player,
                            k,
                            self.explore_eps,
                            self.boltzmann_t,
                            False,
                            sad,
                            self.shuffle_color,
                            hide_action,
                            self.trinary,
                            self.replay_buffer,
                            self.multi_step,
                            self.max_len,
                            self.gamma,
                            self.convention,
                            self.act_parameterized,
                            convention_index,
                            self.convention_act_override[k],
                            self.convention_fict_act_override,
                            self.use_experience[k],
                            self.belief_stats,
                            self.sad_legacy
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
                    convention_index_count += 1

                actors.append(thread_actors)

        self.actors = actors
        print("ActGroup created")

    def start(self):
        for runner in self.model_runners:
            runner.start()

        for runners in self.partner_runners:
            for runner in runners:
                runner.start()

        if self.belief_runner is not None:
            for runner in self.belief_runner:
                runner.start()

    def stop(self):
        for runner in self.model_runners:
            runner.stop()

        for runners in self.partner_runners:
            for runner in runners:
                runner.stop()

        if self.belief_runner is not None:
            for runner in self.belief_runner:
                runner.stop()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)
