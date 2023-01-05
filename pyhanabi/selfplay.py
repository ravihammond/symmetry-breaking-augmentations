# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
import sys
import argparse
import pprint
import json
import wandb
import gc
import pprint
pprint = pprint.pprint
import copy
import numpy as np
import torch
from torch import nn

from act_group import ActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2
import utils

from convention_belief import ConventionBelief
from tools.wandb_logger import log_wandb, log_wandb_test
from google_cloud_handler import GoogleCloudHandler

def selfplay(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    expected_eps = np.mean(explore_eps)
    
    if args.boltzmann_act:
        boltzmann_beta = utils.generate_log_uniform(
            1 / args.max_t, 1 / args.min_t, args.num_t
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))
    else:
        boltzmann_t = []

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.train_bomb,
        args.max_len,
    )

    agent_in_dim = games[0].feature_size(args.sad),
    if args.parameterized:
        agent_in_dim = tuple([x + args.num_parameters for x in agent_in_dim[0]])
    else:
        agent_in_dim = tuple([x for x in agent_in_dim[0]])

    agent = r2d2.R2D2Agent(
        (args.method == "vdn"),
        args.multi_step,
        args.gamma,
        args.eta,
        args.train_device,
        agent_in_dim,
        args.rnn_hid_dim,
        games[0].num_action(),
        args.net,
        args.num_lstm_layer,
        args.boltzmann_act,
        False,  # uniform priority
        args.off_belief,
        parameterized=args.parameterized,
        parameter_type=args.parameter_type,
        num_parameters=args.num_parameters,
        weight_file=args.save_dir
    )
    agent.sync_target_with_online()

    cfgs = {
        "sad": args.sad,
        "hide_action": args.hide_action,
        "shuffle_color": args.shuffle_color,
        "multi_step": args.multi_step,
        "max_len": args.max_len,
        "gamma": args.gamma,
        "parameterized": args.parameterized,
    }

    if args.load_model and args.load_model != "None":
        if args.off_belief and args.belief_model != "None":
            belief_config = utils.get_train_config(args.belief_model)
            args.load_model = belief_config["policy"]

        print("*****loading pretrained model*****")
        print(args.load_model)
        utils.load_weight(agent.online_net, args.load_model, args.train_device)
        print("*****done*****")

    # use clone bot for additional bc loss
    if args.clone_bot and args.clone_bot != "None":
        clone_bot = utils.load_supervised_agent(args.clone_bot, args.train_device)
    else:
        clone_bot = None

    agent = agent.to(args.train_device)
    optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(agent)
    eval_agent = agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    belief_model = None
    belief_config = None
    if args.off_belief and args.belief_model != "None":
        print(f"load belief model from {args.belief_model}")
        from belief_model import ARBeliefModel

        belief_devices = args.belief_device.split(",")
        belief_config = utils.get_train_config(args.belief_model)
        belief_model = []
        for device in belief_devices:
            belief_model.append(ARBeliefModel.load(
                args.belief_model,
                device,
                5,
                args.num_fict_sample,
                belief_config["fc_only"],
                belief_config["parameterized"],
                belief_config["num_parameters"],
                args.belief_model,
            ))

    train_partners = load_partner_agents(
            args, 
            args.train_partner_models, 
            args.train_partner_sad_legacy, 
            "train")
    if args.test_partner_models != "None":
        test_partners = load_partner_agents(
                args, 
                args.test_partner_models, 
                args.test_partner_sad_legacy, 
                "test")
    else: 
        test_partners = None 


    convention = load_json_list(args.convention)

    convention_act_override = [0, args.convention_act_override]
    use_experience = [1, 1 - args.static_partner]

    act_group = ActGroup(
        args.act_device, # devices
        [agent], # agents
        [cfgs], # cfgs
        args.seed, # seed
        args.num_thread, # num_thread
        args.num_player, # num_player
        args.num_game_per_thread, # num_game_per_thread
        [explore_eps], # explore_eps
        [boltzmann_t], # boltzmann_t
        args.method, # method
        True,  # trinary, 3 bits for aux task
        replay_buffer, # replay_buffer
        args.off_belief, # off_belief
        belief_model, # belief_model
        belief_config, # belief_cfg
        convention, # convention
        convention_act_override, # convention_act_overrided
        args.convention_fict_act_override, # convention_fic_act_override
        train_partners, # partners
        args.static_partner, # # static_partner
        use_experience, # use_experience
        args.belief_stats, # belief_stats
        False, # sad_legacy
        runner_div=args.runner_div, # runner_div
        num_parameters=args.num_parameters, # num_parameters
    )

    context, threads = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )

    if args.gcloud_upload:
        gc_save_dir = os.path.basename(args.save_dir)
        gc_handler = GoogleCloudHandler(
            "aiml-reid-research",
            "Ravi",
            "hanabi-conventions/" + gc_save_dir,
            "/app/pyhanabi/exps/" + gc_save_dir,
        )
        gc_handler.assert_directory_doesnt_exist()


    act_group.start()
    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)


    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()

    last_loss = 0


    for epoch in range(args.num_epoch):
        print("beginning of epoch:", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()

        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                act_group.update_model(agent)

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch.time("sample data")

            loss, priority, online_q = agent.loss(batch, args.aux_weight, stat)
            if clone_bot is not None and args.clone_weight > 0:
                bc_loss = agent.behavior_clone_loss(
                    online_q, batch, args.clone_t, clone_bot, stat
                )
                loss = loss + bc_loss * args.clone_weight
            loss = (loss * weight).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            replay_buffer.update_priority(priority)
            stopwatch.time("updating priority")

            last_loss = loss.detach().item()
            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

        count_factor = args.num_player if args.method == "vdn" else 1
        print("epoch: %d" % epoch)
        tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
        stopwatch.summary()
        stat.summary(epoch)

        train_eval_score = run_evaluation(
            args,
            agent,
            eval_agent,
            epoch,
            0, 
            convention,
            convention_act_override,
            train_partners,
            test_partners,
            saver,
            clone_bot,
            act_group,
            last_loss,
        )

        force_save_name = None
        if epoch > 0 and epoch % args.save_checkpoints == 0:
            force_save_name = "model_epoch%d" % epoch
        saver.save(
            None, 
            agent.online_net.state_dict(), 
            train_eval_score, 
            force_save_name=force_save_name,
        )

        if args.gcloud_upload:
            upload_to_google_cloud(args, gc_handler, epoch)

        epoch += 1
        print("==========", flush=True)

def run_evaluation(
    args,
    agent,
    eval_agent,
    epoch,
    explore_eps,
    convention,
    convention_act_override,
    train_partners,
    test_partners,
    saver,
    clone_bot,
    act_group,
    last_loss,
):
    eval_seed = (9917 + epoch * 999999) % 7777777
    eval_agent.load_state_dict(agent.state_dict())
    eval_agents = [eval_agent for _ in range(args.num_player)]

    def eval(partners, convention_act_override, shuffle_colour):
        return evaluate(
            eval_agents,
            args.num_eval_games,
            eval_seed,
            args.eval_bomb,
            explore_eps,
            args.sad,
            args.hide_action,
            device=args.train_device,
            convention=convention,
            override=convention_act_override,
            act_parameterized=[args.parameterized, args.parameterized],
            partners=partners,
            num_parameters=args.num_parameters,
            shuffle_colour=shuffle_colour,
        )

    train_score, train_perfect, train_scores, _, train_eval_actors = eval(
        train_partners, convention_act_override, [args.shuffle_color, 0])

    test_convention_override = [0,0]
    print("epoch %d" % epoch)
    print("train score: %.4f, train perfect: %.2f" % \
            (train_score, train_perfect * 100))

    if args.test_partner_models != "None":
        test_shuffle_colour = [0,0]
        test_score, test_perfect, test_scores, _, test_eval_actors = eval(
            test_partners, test_convention_override, test_shuffle_colour)

        print("test score: %.4f, test perfect: %.2f" % \
                (test_score, test_perfect * 100))

    convention_for_stats = []
    if args.record_convention_stats:
        convention_for_stats = convention

    if args.wandb:
        if args.test_partner_models != "None":
            log_wandb_test(
                train_score, 
                train_perfect, 
                train_scores, 
                train_eval_actors, 
                last_loss, 
                test_score, 
                test_perfect, 
                test_scores, 
                test_eval_actors, 
                convention_for_stats)
        else:
            log_wandb(
                train_score, 
                train_perfect, 
                train_scores, 
                train_eval_actors, 
                last_loss, 
                convention_for_stats)

    if args.test_partner_models != "None":
        return copy.copy(test_score)
    
    return copy.copy(train_score)


def upload_to_google_cloud(args, gc_handler, epoch):
    # Upload log file
    gc_handler.upload_from_file_name("train.log")

    # Upload latest model
    gc_handler.upload_from_file_name("latest.pthw")

    # Upload top k models
    for i in range(5):
        gc_handler.upload_from_file_name(f"model{i}.pthw")

    # Upload forced saved model
    if epoch > 0 and epoch % args.save_checkpoints == 0:
        force_save_name = "model_epoch%d.pthw" % epoch
        gc_handler.upload_from_file_name(force_save_name)


def load_json_list(path):
    print("load_json_list:", path)
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


def load_partner_agents(args, partner_models, sad_legacy, partner_type):
    if partner_models is "None":
        return None

    if type(partner_models) is list:
        model_paths = partner_models
    else:
        print(f"loading {partner_type} agents")
        model_paths = load_json_list(partner_models)

        train_set_indexes = load_json_list(
                args.train_test_splits)[args.split_index][partner_type]
        model_paths = [model_paths[i] for i in train_set_indexes]

    partners = []

    for partner_model_path in model_paths:
        partner_cfg = {
            "sad": False, 
            "hide_action": False,
            "weight": partner_model_path,
            "sad_legacy": False,
        }

        overwrite = {}
        overwrite["vdn"] = False
        overwrite["device"] = "cuda:0"
        overwrite["boltzmann_act"] = False

        if sad_legacy:
            partner_cfg["agent"] = utils.load_sad_model(
                    partner_model_path, args.train_device)
            partner_cfg["sad"] = True
            partner_cfg["hide_action"] = False
            partner_cfg["parameterized"] = False
            partner_cfg["sad_legacy"] = True
            partners.append(partner_cfg)
            continue

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
        partner_cfg["agent"] = partner_agent

        partners.append(partner_cfg)

    return partners

def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--boltzmann_act", type=int, default=0)
    parser.add_argument("--min_t", type=float, default=1e-3)
    parser.add_argument("--max_t", type=float, default=1e-1)
    parser.add_argument("--num_t", type=int, default=80)
    parser.add_argument("--hide_action", type=int, default=0)
    parser.add_argument("--off_belief", type=int, default=0)
    parser.add_argument("--belief_model", type=str, default="None")
    parser.add_argument("--num_fict_sample", type=int, default=10)
    parser.add_argument("--belief_device", type=str, default="cuda:1")

    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--clone_bot", type=str, default="", help="behavior clone loss")
    parser.add_argument("--clone_weight", type=float, default=0.0)
    parser.add_argument("--clone_t", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument(
        "--eta", type=float, default=0.9, help="eta for aggregate priority"
    )
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/ffwd/lstm"
    )

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    parser.add_argument("--save_checkpoints", type=int, default=100)
    parser.add_argument("--convention", type=str, default="None")
    parser.add_argument("--parameterized", type=int, default=0)
    parser.add_argument("--parameter_type", type=str, default="one_hot")
    parser.add_argument("--num_parameters", type=int, default=0)
    parser.add_argument("--no_evaluation", type=int, default=0)
    parser.add_argument("--convention_act_override", type=int, default=0)
    parser.add_argument("--convention_fict_act_override", type=int, default=0)
    parser.add_argument("--train_partner_models", type=str, default="None")
    parser.add_argument("--train_partner_model", type=str, default="None")
    parser.add_argument("--test_partner_models", type=str, default="None")
    parser.add_argument("--test_partner_sad_legacy", type=int, default=0)
    parser.add_argument("--train_partner_sad_legacy", type=int, default=0)
    parser.add_argument("--train_test_splits", type=str, default="None")
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--static_partner", type=int, default=0)
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--gcloud_upload", type=int, default=0)
    parser.add_argument("--belief_stats", type=int, default=0)
    parser.add_argument("--runner_div", type=str, default="duplicated")
    parser.add_argument("--num_eval_games", type=int, default=1000)
    parser.add_argument("--record_convention_stats", type=int, default=0)

    args = parser.parse_args()

    if args.off_belief == 1:
        args.method = "iql"
        args.multi_step = 1
        assert args.net in ["publ-lstm"], "should only use publ-lstm style network"
        assert not args.shuffle_color
    assert args.method in ["vdn", "iql"]

    # Add training split indexes to save directory name
    if args.train_partner_models != "None" and \
            args.train_test_splits != "None":
        indexes = load_json_list(args.train_test_splits)[args.split_index]["train"]
        indexes = [x + 1 for x in indexes]
        args.save_dir = args.save_dir + "_" + '_'.join(str(x) for x in indexes)

    # Single training partner being used
    if args.train_partner_model != "None" and \
            args.train_partner_models == "None":
        args.train_partner_models = [args.train_partner_model]

    # Add pastplay model to save directory
    if args.load_model and args.static_partner:
        model_name = os.path.basename(os.path.dirname(args.load_model))
        args.save_dir = args.save_dir + "_" + model_name
        print(args.save_dir)

    return args

def setup_wandb(args):
    if not args.wandb:
        return 

    run_name = os.path.basename(os.path.normpath(args.save_dir))
    wandb.init(
        project="hanabi-conventions", 
        entity="ravihammond",
        config=args,
        name=run_name,
    )
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    setup_wandb(args)
    selfplay(args)

