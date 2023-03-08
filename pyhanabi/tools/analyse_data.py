import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
pprint = pprint.pprint
import copy

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


def bombout_rate(args):
    df = pd.read_pickle(args.path, compression="gzip")
    # print("\n".join(df.columns))
    # sys.exit()
    # print([c for c in df.columns if "player" in c])

    # df = df[df.turn <= 70]
    plays = df[(df.action >= 5) & (df.action <= 9)]

    colours = ["red", "yellow", "green", "white", "blue"]

    # plays[[]]

    "red_fireworks"
    "card_2_R1_belief"

    # f"card_{action - 5}_{col}{rank}_belief"
    # plays["knowledge_of_played"] =

    # br = df[df.player == "br_sad_six_1_3_6_7_8_12"]
    # sad = df[df.player == "sad_2"]

    # br.groupby("turn")["action_trigger_bomb"].mean().plot(label="br")
    # sad.groupby("turn")["action_trigger_bomb"].mean().plot(label="sad")
    # br.groupby("turn")["action_trigger_bomb"].count().plot()
    # plt.legend()
    # plt.show()


def knowledge_of_played_card(args):
    df = pd.read_pickle(args.path, compression="gzip")
    # df = df[df.game == "br_sad_six_1_2_5_6_11_12_vs_sad_3_game_1"]
    # df = df[df.player == "br_sad_six_1_2_5_6_11_12"]
    # df = df[df.turn == "67"]
    df = df[(df.action >= 5) & (df.action <= 9)]
    conditions = []
    choices = []
    colours = "RYGWB"
    colours_str = ["red", "yellow", "green", "white", "blue"]

    def firework_conditions(card, condition_prev, choice_prev, colour):
        for rank in range(6):
            condition = copy.copy(condition_prev)
            choice = copy.copy(choice_prev)
            condition &= df[f"{colours_str[colour]}_fireworks"] == rank
            if rank < 5: 
                choice += df[f"card_{card}_{colours[colour]}{rank + 1}_belief"] 

            if colour < 4:
                firework_conditions(card, condition, choice, colour + 1)
            else:
                conditions.append(condition)
                choices.append(choice)

    for card in range(5):
        condition = df["action"] == 5 + card
        choice = 0
        firework_conditions(card, condition, choice, 0)

    df["knowledge_of_played"] = np.select(conditions, choices, default=-1)
    df = df[(df.action_trigger_bomb == 1)
            & (df.knowledge_of_played == 0)]

    columns = [
        # "game",
        "player",
        "partner",
        "turn",
        "action",
        "card_4",
        "red_fireworks",
        "yellow_fireworks",
        "green_fireworks",
        "white_fireworks",
        "blue_fireworks",
        "card_4_R5_belief",
        "card_4_Y3_belief",
        "card_4_G3_belief",
        "card_4_W4_belief",
        "card_4_B5_belief",
        # "card_4_R_hinted",
        # "card_4_Y_hinted",
        # "card_4_G_hinted",
        # "card_4_W_hinted",
        # "card_4_B_hinted",
        # "card_4_1_hinted",
        "card_4_2_hinted",
        # "card_4_3_hinted",
        # "card_4_4_hinted",
        # "card_4_5_hinted",
        "knowledge_of_played",
        "action_trigger_bomb",
    ]

    # pprint(list(df.columns.values))
    print(df[columns].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    knowledge_of_played_card(args)

