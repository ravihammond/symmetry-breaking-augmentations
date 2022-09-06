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
        convention,
        act_parameterized,
        belief_parameterized,
        convention_act_override,
        convention_fict_act_override,
        partner_agent,
        partner_cfg,
        static_partner,
        use_experience,
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
        for i, dev in enumerate(self.devices):
            runner = rela.BatchRunner(agent.clone(dev), dev)
            runner.add_method("act", 5000)
            runner.add_method("compute_priority", 100)
            if off_belief:
                runner.add_method("compute_target", 5000)
            self.model_runners.append(runner)
        self.num_runners = len(self.model_runners)

        self.partner_runners = []
        if static_partner:
            for i, dev in enumerate(self.devices):
                runner = rela.BatchRunner(partner_agent.clone(dev), dev)
                runner.add_method("act", 5000)
                self.partner_runners.append(runner)

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
        self.act_parameterized = act_parameterized
        self.belief_parameterized = belief_parameterized
        self.convention_act_override = convention_act_override
        self.convention_fict_act_override = convention_fict_act_override
        self.partner_agent = partner_agent
        self.partner_cfg = partner_cfg
        self.static_partner = static_partner
        self.use_experience = use_experience

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
                        0, # belief paramaterized
                        0, # convention index
                        0, # convention act override
                        0, # convention fict act override
                        1, # convention use experience
                    )
                    self.seed += 1
                    thread_actors.append([actor])
                actors.append(thread_actors)
        elif self.method == "iql":
            for i in range(self.num_thread):
                thread_actors = []
                for j in range(self.num_game_per_thread):
                    game_actors = []
                    convention_index = -1 if len(self.convention) == 0 \
                            else convention_index_count % len(self.convention)
                    for k in range(self.num_player):
                        if k > 0 and self.static_partner:
                            self.model_runners = self.partner_runners
                            self.sad = self.partner_cfg["sad"]
                            self.hide_action = self.partner_cfg["hide_action"]

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
                            self.act_parameterized,
                            self.belief_parameterized,
                            convention_index,
                            self.convention_act_override[k],
                            self.convention_fict_act_override,
                            self.use_experience[k],
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

        for runner in self.partner_runners:
            runner.start()

        if self.belief_runner is not None:
            for runner in self.belief_runner:
                runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)
