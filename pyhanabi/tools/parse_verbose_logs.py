import argparse
import sys
import pprint
import re
pprint = pprint.pprint

GAME_STATS = [
    "epoch",
    "score",
    "perfect",
    "bomb_out_rate",
    "non_zero_mean",
]

ACTOR_STATS = [
    "total_cards_played",
    "card_played_knowledge_none",
    "card_played_knowledge_color",
    "card_played_knowledge_rank",
    "card_played_knowledge_both",
    "play",
    "discard",
    "hint_colour",
    "hint_red",
    "hint_yellow",
    "hint_green",
    "hint_white",
    "hint_blue",
    "hint_rank",
    "hint_1",
    "hint_2",
    "hint_3",
    "hint_4",
    "hint_5",
    "convention_available",
    "convention_played",
    "convention_played_correct",
    "convention_played_incorrect",
    "convention_played_0_playable",
    "convention_played_1_playable",
    "convention_played_2_playable",
    "convention_played_3_playable",
    "convention_played_4_playable",
]


def parse_logs(filename, max_epochs):
    lines = open(filename, "r").readlines()
    stats = initialise_stats()

    for i, l in enumerate(lines):
        if len(stats["epoch"]) > max_epochs:
            break
        parse_game_stat(stats, l, i)
        parse_actor_stat(stats, l, i, 0)
        parse_actor_stat(stats, l, i, 1)

    return stats

    
def initialise_stats():
    stats = {}
    for stat in GAME_STATS:
        stats[stat] = []
    for stat in ACTOR_STATS:
        for i in range(2):
            stats[f"actor{i}_" + stat] = []
    return stats


def parse_game_stat(stats, line, i):
    for stat in GAME_STATS:
        if stat + ':' in line:
            value = extract_value_from_line(line, stat)
            stats[stat].append(value)


def parse_actor_stat(stats, line, i, actor):
    for stat in ACTOR_STATS:
        stat_name = f"actor{actor}_" + stat
        if stat_name + ':' in line:
            value = extract_value_from_line(line, stat_name)
            stats[stat_name].append(value)


def extract_value_from_line(line, stat):
    tokens = line.split()
    stat_index = [i for i, s in enumerate(tokens) if stat + ':' in s][0]
    value_str = tokens[stat_index + 1]
    return float(re.findall(r'\d+(?:\.\d+)?', value_str)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=float("inf"))
    args = parser.parse_args()
    parse_logs(args.filename, args.max_epochs)

