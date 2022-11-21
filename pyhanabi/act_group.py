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
        agents,
        cfgs,
        seed,
        num_thread,
        num_player,
        num_game_per_thread,
        explore_eps,
        boltzmann_t,
        method,
        trinary,
        replay_buffer,
        off_belief,
        belief_model,
        convention,
        convention_act_override,
        convention_fict_act_override,
        partner_agents,
        partner_cfgs,
        static_partner,
        use_experience,
        belief_stats,
        sad_legacy,
        *,
        runner_div="duplicated",
        num_parameters=0,
    ):
        self.devices = devices.split(",")
        self.method = method
        self.seed = seed
        self.num_thread = num_thread
        self.num_player = num_player
        self.num_game_per_thread = num_game_per_thread
        self.explore_eps = explore_eps
        self.boltzmann_t = boltzmann_t
        self.trinary = trinary
        self.replay_buffer = replay_buffer
        self.convention = convention
        self.convention_act_override = convention_act_override
        self.convention_fict_act_override = convention_fict_act_override
        self.partner_cfgs = partner_cfgs
        self.static_partner = static_partner
        self.use_experience = use_experience
        self.belief_stats = belief_stats
        self.sad_legacy = sad_legacy
        self.cfgs = cfgs
        self.off_belief = off_belief
        self.belief_model = belief_model
        self.belief_runner = None
        self.runner_div = runner_div
        self.num_parameters = num_parameters
        if self.sad_legacy:
            self.trinary = True
        self.num_agents = len(agents)

        (self.model_runners, self.belief_runner, self.partner_runners) = \
                self.load_runners( 
            agents, self.devices, num_thread, off_belief, runner_div, belief_model)

        self.num_runners = len(self.model_runners)

        self.actors = self.create_r2d2_actors()

    def load_runners(self, agents, devices, num_thread, off_belief, 
            runner_div, belief_model):
        model_runners = []
        belief_runner = None
        partner_runners = []

        def create_runner(agent, dev, off_belief):
            runner = rela.BatchRunner(agent.clone(dev), dev)
            runner.add_method("act", 5000)
            runner.add_method("compute_priority", 100)
            if off_belief:
                runner.add_method("compute_target", 5000)
            return runner

        if runner_div == "duplicated":
            for dev in devices:
                for agent in agents:
                    runner = create_runner(agent, dev, off_belief)
                    model_runners.append(runner)

        elif runner_div == "round_robin":
            for i, agent in enumerate(agents):
                dev = devices[i % len(devices)]
                runner = create_runner(agent, dev, off_belief)
                model_runners.append(runner)

        if belief_model is not None:
            for bm in belief_model:
                belief_runner = rela.BatchRunner(bm, bm.device, 5000, ["sample"])

        return model_runners, belief_runner, partner_runners

    def create_r2d2_actors(self):
        partner_idx = -1
        parameter_index = 0

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
            for t_idx in range(self.num_thread):
                thread_actors = []
                agent_idx = t_idx % self.num_agents
                runner_idx = t_idx % (self.num_agents * len(self.devices))
                if self.runner_div == "round_robin":
                    runner_idx = t_idx % self.num_agents

                for g_idx in range(self.num_game_per_thread):
                    game_actors = []

                    if self.num_parameters > 0:
                        parameter_index = (parameter_index + 1) % self.num_parameters

                    if len(self.partner_runners) > 0:
                        partner_idx = (partner_idx + 1) % len(self.partner_runners)

                    for k in range(self.num_player):
                        runner = self.model_runners[runner_idx]
                        sad = self.cfgs[agent_idx]["sad"]
                        hide_action = self.cfgs[agent_idx]["hide_action"]
                        weight = "cosca"
                        if k > 0 and self.static_partner:
                            runner = self.partner_runners[t_idx % self.num_runners][partner_idx]
                            sad = self.partner_cfgs[partner_idx]["sad"]
                            hide_action = self.partner_cfgs[partner_idx]["hide_action"]
                            weight = self.partner_cfgs[partner_idx]["weight"]

                        print("loading runner to actor, parameter:", parameter_index)
                        runner.print_model()

                        actor = hanalearn.R2D2Actor(
                            runner,
                            self.seed,
                            self.num_player,
                            k,
                            self.explore_eps[agent_idx],
                            self.boltzmann_t[agent_idx],
                            False,
                            sad,
                            self.cfgs[agent_idx]["shuffle_color"],
                            hide_action,
                            self.trinary,
                            self.replay_buffer,
                            self.cfgs[agent_idx]["multi_step"],
                            self.cfgs[agent_idx]["max_len"],
                            self.cfgs[agent_idx]["gamma"],
                            self.convention,
                            self.cfgs[agent_idx]["parameterized"],
                            parameter_index,
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
                                print("loading belief runner to actor")
                                self.belief_runner.print_model()
                                actor.set_belief_runner(self.belief_runner)
                        self.seed += 1
                        game_actors.append(actor)

                    for k in range(self.num_player):
                        partners = game_actors[:]
                        partners[k] = None
                        game_actors[k].set_partners(partners)

                    thread_actors.append(game_actors)
                    parameter_index += 1

                actors.append(thread_actors)

        return actors

    def start(self):
        for runner in self.model_runners:
            runner.start()

        for runners in self.partner_runners:
            for runner in runners:
                runner.start()

        if self.belief_runner is not None:
            self.belief_runner.start()

    def stop(self):
        for runner in self.model_runners:
            runner.stop()

        for runners in self.partner_runners:
            for runner in runners:
                runner.stop()

        if self.belief_runner is not None:
            self.belief_runner.stop()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)
