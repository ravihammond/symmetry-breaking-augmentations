import sys
import os
import argparse
import pprint
pprint = pprint.pprint
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from create import *
import rela
import r2d2
import utils

np.set_printoptions(threshold=sys.maxsize, linewidth=10000)

CARD_ID_TO_STRING = np.array([
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "Y1",
    "Y2",
    "Y3",
    "Y4",
    "Y5",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
])

ACTION_ID_TO_STRING = np.array([
    "Discard 0",
    "Discard 1",
    "Discard 2",
    "Discard 3",
    "Discard 4",
    "Play 0",
    "Play 1",
    "Play 2",
    "Play 3",
    "Play 4",
    "Reveal color R",
    "Reveal color Y",
    "Reveal color G",
    "Reveal color W",
    "Reveal color B",
    "Reveal rank 1",
    "Reveal rank 2",
    "Reveal rank 3",
    "Reveal rank 4",
    "Reveal rank 5",
    "INVALID"
])

ACTION_ID_TO_STRING_SHORT = np.array([
    "discard_0",
    "discard_1",
    "discard_2",
    "discard_3",
    "discard_4",
    "play_0",
    "play_1",
    "play_2",
    "play_3",
    "play_4",
    "hint_R",
    "hint_Y",
    "hint_G",
    "hint_W",
    "hint_B",
    "hint_1",
    "hint_2",
    "hint_3",
    "hint_4",
    "hint_5",
    "INVALID"
])


def save_games(args):
    now = datetime.now()

    # Load agents
    agents = load_agents(args)

    # Get replay buffer of games
    print("generating data")
    replay_buffer = generate_replay_data(args, agents)

    # Convert to dataframe
    print("extracting dataframe")
    data = replay_to_dataframe(args, replay_buffer, now)

    if args.save:
        save_all_data(args, data, now)


def load_agents(args):
    weights = [args.weight1, args.weight2]
    agents = []
    for i, weight in enumerate(weights):
        agents.append(load_agent(weight, args.sad_legacy[i], args.device))
    return agents


def load_agent(weight, sad_legacy, device):
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
        agent = utils.load_sad_model(weight, device)
        cfg = default_cfg
    else:
        agent, cfg = utils.load_agent(weight, {
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
    agents,
):
    seed = args.seed
    num_player = 2
    num_thread = args.num_thread
    if args.num_game < num_thread:
        num_thread = args.num_game

    runners = []
    for agent, _, _, _ in agents:
        runner = rela.BatchRunner(agent.clone(args.device), args.device)
        runner.add_method("act", 5000)
        runner.add_method("compute_priority", 100)
        runners.append(runner)

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
                cfgs = agents[i][1]
                actor = hanalearn.R2D2Actor(
                    runners[i], # runner
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
                    agents[i][3], # sadLegacy
                    False, # beliefSadLegacy
                    False, # colorShuffleSync
                )

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

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    return replay_buffer


def replay_to_dataframe(args, replay_buffer, now):
    date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

    assert(replay_buffer.size() % args.batch_size == 0)
    num_batches = (int)(replay_buffer.size() / args.batch_size)

    for batch_index in range(num_batches):
        range_start = batch_index * args.batch_size
        range_end = batch_index * args.batch_size + args.batch_size
        sample_id_list = [*range(range_start, range_end, 1)]

        batch1, batch2 = replay_buffer.sample_from_list_split(
                args.batch_size, "cpu", sample_id_list)
        
        data = batch_to_dataset(args, batch1, batch2, date_time)

    return data


def batch_to_dataset(args, batch1, batch2, date_time):
    df = pd.DataFrame()

    obs_df = player_dataframe(args, batch1, 0, date_time)
    df = pd.concat([df, obs_df])

    obs_df = player_dataframe(args, batch2, 1, date_time)
    df = pd.concat([df, obs_df])

    df = df.reset_index(drop=True)

    if args.verbose:
        print("num cows:", df.shape[0])
        print("num columns:", len(list(df.columns.values)))

    return df

def player_dataframe(args, batch, player, date_time):
    df = pd.DataFrame()

    # Add meta data
    meta_df = meta_data(args, batch, player, date_time)
    df = pd.concat([df, meta_df])

    # Add turn numbers
    hand_df = turn_data(args, batch)
    df = pd.concat([df, hand_df], axis=1)

    # Add observation
    obs_df = extract_obs(args, batch.obs, player)
    df = pd.concat([df, obs_df], axis=1)

    # Add legal moves
    legal_moves_df = extract_legal_moves(args, batch.obs["legal_move"])
    df = pd.concat([df, legal_moves_df], axis=1)

    # Add Action
    action_df = extract_action(args, batch.action["a"])
    df = pd.concat([df, action_df], axis=1)

    # Add Q Values
    action_df = extract_q_values(args, batch.action["all_q"])
    df = pd.concat([df, action_df], axis=1)

    # Add Terminal
    terminal_df = extract_terminal(args, batch.terminal)
    df = pd.concat([df, terminal_df], axis=1)

    # Add bombs triggered
    df = add_bombs_triggered(args, df)

    # Remove rows after game has ended
    df = remove_states_after_terminal(args, df, batch.terminal)

    return df


def meta_data(args, batch, player, date_time):
    priv_s = batch.obs["priv_s"]
    num_rows = priv_s.shape[0] * priv_s.shape[1]

    game_names = []

    for i in range(priv_s.shape[1]):
        game_names.append(f"{args.player_name[0]}_vs_{args.player_name[1]}_game_{i}")

    data = np.array(game_names, )
    data = np.repeat(data, priv_s.shape[0])
    data = np.reshape(data, (num_rows, 1))

    meta_data = np.array([
        args.player_name[player],
        args.player_name[(player + 1) % 2],
        args.data_type,
        date_time
    ], dtype=str)

    meta_data = np.tile(meta_data, (num_rows, 1))
    data = np.concatenate((data, meta_data), axis=1)

    labels = [
        "game",
        "player",
        "partner",
        "data_type",
        "datetime",
    ]

    return pd.DataFrame(
        data=data,
        columns=labels
    )


def turn_data(args, batch):
    shape = batch.obs["priv_s"].shape
    data = np.arange(0,80, dtype=np.uint8)
    data = np.tile(data, (shape[1], 1))
    data = np.reshape(data, (shape[0] * shape[1],))
    labels = ["turn"]

    return pd.DataFrame(
        data=data,
        columns=labels
    )


def extract_obs(args, obs, player):
    df = pd.DataFrame()

    if args.sad_legacy[player]:
        own_hand_str = "own_hand_ar"
        # Make sad priv_s the same as OBL priv_s
        priv_s = obs["priv_s"][:, :, 125:783]
    else:
        own_hand_str = "own_hand"
        priv_s = obs["priv_s"]

    partner_hand_idx = 125
    missing_cards_idx = 127
    board_idx = 203
    discard_idx = 253
    last_action_idx = 308
    v0_belief_idx = 658

    # Own hand
    hand_df = extract_hand(args, obs[own_hand_str], "")
    df = pd.concat([df, hand_df], axis=1)

    # Partner Hand
    partner_hand = np.array(priv_s[:, :, :partner_hand_idx])
    hand_df = extract_hand(args, partner_hand, "partner_")
    df = pd.concat([df, hand_df], axis=1)

    # Hands missing Card
    missing_cards = np.array(priv_s[:, :, partner_hand_idx:missing_cards_idx])
    missing_cards_df = extract_missing_cards(args, missing_cards)
    df = pd.concat([df, missing_cards_df], axis=1)

    # Board
    board = np.array(priv_s[:, :, missing_cards_idx:board_idx])
    board_df = extract_board(args, board)
    df = pd.concat([df, board_df], axis=1)

    # Discards
    discards = np.array(priv_s[:, :, board_idx:discard_idx])
    discards_df = extract_discards(args, discards)
    df = pd.concat([df, discards_df], axis=1)

    # Last Action
    last_action = np.array(priv_s[:, :, discard_idx:last_action_idx])
    last_action_df = extract_last_action(args, last_action)
    df = pd.concat([df, last_action_df], axis=1)

    # Knowledge
    card_knowledge = np.array(priv_s[:, :, last_action_idx:v0_belief_idx])
    card_knowledge_df = extract_card_knowledge(args, card_knowledge)
    df = pd.concat([df, card_knowledge_df], axis=1)

    return df


def extract_hand(args, hand, label_str):
    hand = np.array(hand, dtype=int)
    shape = hand.shape
    hand = np.reshape(hand, (shape[0], shape[1], 5, 25))
    hand = np.swapaxes(hand, 0, 1) 
    cards = np.argmax(hand, axis=3)
    cards = np.reshape(cards, (cards.shape[0] * cards.shape[1], 5))
    cards = cards.astype(np.uint8)

    labels = []
    for i in range(5):
        labels.append(f"{label_str}card_{i}")

    # cards = CARD_ID_TO_STRING[cards]

    return pd.DataFrame(
        data=cards,
        columns=labels
    )


def extract_missing_cards(args, missing_cards):
    missing_cards = np.array(missing_cards, dtype=np.uint8)
    missing_cards = np.swapaxes(missing_cards, 0, 1)
    num_rows = missing_cards.shape[0] * missing_cards.shape[1]
    missing_cards = np.reshape(missing_cards, (num_rows, missing_cards.shape[2]))

    labels = ["own_missing_card", "partner_missing_card"]

    return pd.DataFrame(
        data=missing_cards,
        columns=labels
    )

def extract_board(args, board):
    num_rows = board.shape[0] * board.shape[1]
    board = np.array(board, dtype=np.uint8)
    board = np.swapaxes(board, 0, 1)

    # Encoding positions
    deck_idx = 40
    fireworks_idx = 65
    info_idx = 73
    life_idx = 76

    board_data = np.empty((num_rows, 0), dtype=np.uint8)

    # Deck
    deck = board[:, :, :deck_idx]
    deck_size = deck.sum(axis=2)
    deck_size = np.expand_dims(deck_size, axis=2)
    deck_size = np.reshape(deck_size, (num_rows, deck_size.shape[2]))
    board_data = np.concatenate((board_data, deck_size), axis=1)

    # Fireworks
    fireworks = board[:, :, deck_idx:fireworks_idx]
    fireworks = np.reshape(fireworks, (fireworks.shape[0], fireworks.shape[1], 5, 5))
    non_empty_piles = np.sum(fireworks, axis=3)
    empty_piles = non_empty_piles ^ (non_empty_piles & 1 == non_empty_piles)
    fireworks = np.argmax(fireworks, axis=3) + 1 - empty_piles
    fireworks = np.reshape(fireworks, (num_rows, fireworks.shape[2]))
    fireworks = fireworks.astype(np.uint8)
    board_data = np.concatenate((board_data, fireworks), axis=1)

    # Info Tokens
    info = board[:, :, fireworks_idx:info_idx]
    info_tokens = info.sum(axis=2)
    info_tokens = np.expand_dims(info_tokens, axis=2)
    info_tokens = np.reshape(info_tokens, (num_rows, info_tokens.shape[2]))
    board_data = np.concatenate((board_data, info_tokens), axis=1)

    # Life Tokens
    lives = board[:, :, info_idx:life_idx]
    lives = lives.sum(axis=2)
    lives = np.expand_dims(lives, axis=2)
    lives = np.reshape(lives, (num_rows, lives.shape[2]))
    board_data = np.concatenate((board_data, lives), axis=1)

    # Column labels
    labels = ["deck_size"]
    for colour in ["red", "yellow", "green", "white", "blue"]:
        labels.append(f"{colour}_fireworks")
    labels.extend(["info_tokens", "lives"])

    return pd.DataFrame(
        data=board_data,
        columns=labels
    )


def extract_discards(args, discards):
    num_rows = discards.shape[0] * discards.shape[1]
    discards = np.array(discards, dtype=np.uint8)
    discards = np.swapaxes(discards, 0, 1)
    discards_data = np.empty((num_rows, 0), dtype=np.uint8)

    idx_pos_per_rank = [3, 5, 7, 9, 10]
    num_cards_per_rank = [3, 2, 2, 2, 1]
    colours = ["red", "yellow", "green", "white", "blue"]

    bits_per_colour = 10

    labels = []

    for i, colour in enumerate(["red", "yellow", "green", "white", "blue"]):
        offset = i * bits_per_colour

        for j in range(5):
            labels.append(f"{colour}_{j + 1}_discarded")

            end_pos = offset + idx_pos_per_rank[j]
            start_pos = end_pos - num_cards_per_rank[j]
            num_discards = discards[:, :, start_pos:end_pos]
            num_discards = np.sum(num_discards, axis=2)
            num_discards = np.expand_dims(num_discards, axis=2)
            num_discards = np.reshape(num_discards, (num_rows, num_discards.shape[2]))
            discards_data = np.concatenate((discards_data, num_discards), axis=1)

    return pd.DataFrame(
        data=discards_data,
        columns=labels
    )

def extract_last_action(args, last_action):
    num_rows = last_action.shape[0] * last_action.shape[1]
    last_action = np.array(last_action, dtype=np.uint8)
    last_action = np.swapaxes(last_action, 0, 1)

    acting_player_idx = 2
    move_type_idx = 6
    target_player_idx = 8
    colour_revealed_idx = 13
    rank_revealed_idx = 18
    reveal_outcome_idx = 23
    card_position_idx = 28
    card_played_idx = 53
    card_played_scored_idx = 54

    move_type = last_action[:, :, acting_player_idx:move_type_idx]
    card_position = last_action[:, :, reveal_outcome_idx:card_position_idx]
    colour_revealed = last_action[:, :, target_player_idx:colour_revealed_idx]
    rank_revealed = last_action[:, :, colour_revealed_idx:rank_revealed_idx]
    card_played_scored = last_action[:, :, card_played_idx:card_played_scored_idx]

    action_index = [1,0,2,3]
    move_index = range(5)
    action_functions = [card_position, card_position, colour_revealed, rank_revealed]

    conditions = []
    for action_i in action_index:
        for move_i in move_index:
            conditions.append((move_type[:, :, action_i] == 1) & \
                              (action_functions[action_i][:, :, move_i] == 1))
    conditions.append(True)

    move_id = range(21)
    last_action_data = np.select(conditions, move_id, default=20)
    last_action_data = np.expand_dims(last_action_data, axis=2)


    last_action_data = np.concatenate((last_action_data, card_played_scored), axis=2)
    last_action_data = np.reshape(last_action_data, (num_rows, last_action_data.shape[2]))

    return pd.DataFrame(
        data=last_action_data,
        columns=["last_action", "last_action_scored"]
    )


def extract_card_knowledge(args, card_knowledge):
    num_rows = card_knowledge.shape[0] * card_knowledge.shape[1]
    card_knowledge = np.array(card_knowledge)
    card_knowledge = np.swapaxes(card_knowledge, 0, 1)
    card_knowledge = np.reshape(card_knowledge, (num_rows, card_knowledge.shape[2]))

    possible_cards_len = 25
    colour_hinted_len = 5
    rank_hinted_len = 5
    card_len = possible_cards_len + colour_hinted_len + rank_hinted_len
    player_len = card_len * 5

    labels = []

    players = ["", "partner_"]
    colours = "RYGWB"

    for player in range(2):
        for card in range(5):
            for colour in range(5):
                for rank in range(5):
                    labels.append(f"{players[player]}card_{card}_{colours[colour]}{rank+1}_belief")

            for colour in range(5):
                labels.append(f"{players[player]}card_{card}_{colours[colour]}_hinted")

            for rank in range(5):
                labels.append(f"{players[player]}card_{card}_{rank + 1}_hinted")


    return pd.DataFrame(
        data=card_knowledge,
        columns=labels
    )


def extract_legal_moves(args, legal_move):
    num_rows = legal_move.shape[0] * legal_move.shape[1]
    legal_move = np.array(legal_move, dtype=np.uint8)
    legal_move = np.swapaxes(legal_move, 0, 1)
    legal_move = np.reshape(legal_move, (num_rows, legal_move.shape[2]))

    labels=[]

    for move_id in range(21):
        labels.append(f"legal_move_{ACTION_ID_TO_STRING_SHORT[move_id]}")

    df = pd.DataFrame(
        data=legal_move,
        columns=labels
    )

    return df

def extract_action(args, action):
    num_rows = action.shape[0] * action.shape[1]
    action = np.array(action, dtype=np.uint8)
    action = np.swapaxes(action, 0, 1)
    action = np.expand_dims(action, axis=2)
    action = np.reshape(action, (num_rows, action.shape[2]))

    return pd.DataFrame(
        data=action,
        columns=["action"]
    )


def extract_q_values(args, q_values):
    num_rows = q_values.shape[0] * q_values.shape[1]
    q_values = np.array(q_values)
    q_values = np.swapaxes(q_values, 0, 1)
    q_values = np.reshape(q_values, (num_rows, q_values.shape[2]))

    labels = []
    for move_id in range(21):
        labels.append(f"q_value_move_{ACTION_ID_TO_STRING_SHORT[move_id]}")

    return pd.DataFrame(
        data=q_values,
        columns=labels
    )


def extract_terminal(args, terminal):
    num_rows = terminal.shape[0] * terminal.shape[1]
    terminal = np.array(terminal, dtype=np.uint8)
    terminal = np.swapaxes(terminal, 0, 1)
    terminal = np.expand_dims(terminal, axis=2)
    terminal = np.reshape(terminal, (num_rows, terminal.shape[2]))

    return pd.DataFrame(
        data=terminal,
        columns=["terminal"]
    )


def add_bombs_triggered(args, df):
    action = df["action"]
    cards = np.array([ df[f"card_{i}"] for i in range(5) ])
    card_to_colour = np.repeat(np.arange(0,5),5)
    colours = ["red", "yellow", "green", "white", "blue"]
    colour_to_fireworks = np.array([ df[f"{colours[i]}_fireworks"] for i in range(5) ])
    card_to_rank = np.array(list(np.arange(1,6)) * 5)

    condition = []
    for card_position in range(5):
        for colour in range(5):
            condition.append(
                (action == 5 + card_position)
                & (card_to_colour[cards[card_position]] == colour) 
                & (colour_to_fireworks[colour] + 1 != card_to_rank[cards[card_position]]),
            )

    result = [1] * len(condition)

    last_action_data = np.select(condition, result, default=0)

    bombs_triggered_df = pd.DataFrame(
        data=last_action_data,
        columns=["action_trigger_bomb"],
    )
    df = pd.concat([df, bombs_triggered_df], axis=1)

    df["last_action_trigger_bomb"] = np.where(
        (df["last_action"] >= 5)
        & (df["last_action"] <= 9)
        & (df["last_action_scored"] == 0), 1, 0
    )

    return df

def remove_states_after_terminal(args, df, terminal):
    terminal =  np.array(terminal, dtype=np.uint8)
    terminal = np.swapaxes(terminal, 0, 1)
    terminal = np.expand_dims(terminal, axis=2)
    inv_terminal = terminal ^ (terminal & 1 == terminal)
    sum = np.sum(inv_terminal, axis=1)
    rows = np.array(range(sum.shape[0]))
    rows = np.expand_dims(rows, axis=1)
    sumrows = np.hstack((rows, sum))
    sumrows = sumrows.astype(int)
    sumrows = sumrows[sumrows[:,1] < terminal.shape[1]]
    terminal[sumrows[:,0], sumrows[:,1], 0] = 0
    num_rows = terminal.shape[0] * terminal.shape[1]
    remove_rows = np.reshape(terminal, (num_rows, terminal.shape[2]))
    remove_rows = remove_rows.astype(bool)

    remove_rows_df = pd.DataFrame(
        data=remove_rows,
        columns=["remove_rows"],
    )
    df = pd.concat([df, remove_rows_df], axis=1)
    df = df[~df.remove_rows]
    df = df.drop("remove_rows", axis=1)
    return df


def save_all_data(args, data, now):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    date_time = now.strftime("%m.%d.%Y_%H:%M:%S")

    filename = f"{args.player_name[0]}_vs_{args.player_name[1]}_{date_time}.pkl"
    filepath = os.path.join(args.out, filename)

    print("Saving:", filepath)
    data.to_pickle(filepath, compression="gzip")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight1", type=str, required=True)
    parser.add_argument("--weight2", type=str, required=True) 
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--player_name", type=str, required=True) 
    parser.add_argument("--data_type", type=str, required=True) 
    parser.add_argument("--sad_legacy", type=str, default="0,0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    args = parser.parse_args()

    # Convert sad_legacy to valid list of ints
    args.sad_legacy = [int(x) for x in args.sad_legacy.split(",")]
    assert(len(args.sad_legacy) <= 2)
    if (len(args.sad_legacy) == 1):
        args.sad_legacy *= 2

    # batch size is double the number of games
    if args.batch_size is None:
        args.batch_size = args.num_game * 2

    args.player_name = args.player_name.split(",")

    return args

if __name__ == "__main__":
    args = parse_args()
    save_games(args)

