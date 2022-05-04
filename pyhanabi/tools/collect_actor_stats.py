import numpy as np
from collections import defaultdict
import pprint
pprint = pprint.pprint

CARD_INDEX_MAP = ["0", "1", "2", "3", "4"]
COLOUR_MOVE_MAP = ["red", "yellow", "green", "white", "blue"]
RANK_MOVE_MAP = ["1", "2", "3", "4", "5"]

def collect_stats(score, perfect, scores, actors, conventions):
    stats = defaultdict(int)
    record_scores(stats, score, perfect, scores)
    convention_strings = extract_convention_strings(conventions)

    for i, actor in enumerate(actors):
        convention_index = actor.get_convention_index()
        convention_str = convention_strings[convention_index]
        actor_stats = defaultdict(int, actor.get_stats())
        
        record_actor_stats(stats, actor_stats, convention_str, i % 2)

    evaluate_percentages(stats, convention_strings)
    return stats


def record_scores(stats, score, perfect, scores):
    non_zero_scores = [s for s in scores if s > 0]
    non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
    bomb_out_rate = 100 * (1 - len(non_zero_scores) / len(scores))

    stats["score"] = score
    stats["perfect"] = perfect
    stats["non_zero_mean"] = non_zero_mean
    stats["bomb_out_rate"] = bomb_out_rate


def extract_convention_strings(conventions):
    convention_strs = []

    for convention in conventions:
        convention_str = ""
        for i, two_step in enumerate(convention):
            if i > 0:
                convention_str + '-'
            convention_str += two_step[0] + two_step[1]
        convention_strs.append(convention_str)

    return convention_strs


def record_actor_stats(stats, actor_stats, convention_str, player):
    move_stats(stats, actor_stats, player)
    convention_stats(stats, actor_stats, convention_str, player)

 
def move_stats(stats, actor_stats, player):
    move_type_stats(stats, actor_stats, player, CARD_INDEX_MAP, "play")
    move_type_stats(stats, actor_stats, player, CARD_INDEX_MAP, "discard")
    move_type_stats(stats, actor_stats, player, COLOUR_MOVE_MAP, "hint", "_colour")
    move_type_stats(stats, actor_stats, player, RANK_MOVE_MAP, "hint", "_rank")


def move_type_stats(stats, actor_stats, player, move_map, 
        move_type, move_type_suffix=""):
    actor_str = f"actor{player}"
    move_with_suffix = f"{move_type}{move_type_suffix}"

    stats[f"{actor_str}_{move_with_suffix}"] += int(actor_stats[move_with_suffix])

    for move in move_map:
        stats[f"{actor_str}_{move_type}_{move}"] += \
                int(actor_stats[f"{move_type}_{move}"])


def convention_stats(stats, actor_stats, convention, player):
    convention_string = f"convention_{convention}"
    available = f"{convention_string}_available"
    played = f"{convention_string}_played"
    played_correct = f"{convention_string}_played_correct"
    played_incorrect = f"{convention_string}_played_incorrect"

    stats[f"actor{player}_{available}"] += int(actor_stats[available])
    stats[f"actor{player}_{played}"] += int(actor_stats[played])
    stats[f"actor{player}_{played_correct}"] += int(actor_stats[played_correct])
    stats[f"actor{player}_{played_incorrect}"] += int(actor_stats[played_incorrect])


def evaluate_percentages(stats, convention_strings):
    for player in range(2):
        move_percentages(stats, player)
        convention_percentages(stats, convention_strings, player)

def move_percentages(stats, player):
    percent_move_type(stats, player, CARD_INDEX_MAP, "play")
    percent_move_type(stats, player, CARD_INDEX_MAP, "discard")
    percent_move_type(stats, player, COLOUR_MOVE_MAP, "hint", "_colour")
    percent_move_type(stats, player, RANK_MOVE_MAP, "hint", "_rank")

def percent_move_type(stats, player, move_map, move_type, move_type_suffix=""):
    actor_str = f"actor{player}"
    move_with_suffix = f"{move_type}{move_type_suffix}"

    total = stats[f"{actor_str}_{move_with_suffix}"]

    for move in move_map:
        move_type_with_move = f"{actor_str}_{move_type}_{move}"
        move_count = stats[move_type_with_move]
        percentage = percent(move_count, total)
        stats[f"{move_type_with_move}%"] = percentage

def convention_percentages(stats, convention_strings, player):
    actor_str = f"actor{player}"
    for convention_string in convention_strings:
        conv_str = "convention_" + convention_string

        available = stats[f"{actor_str}_{conv_str}_available"]
        played = stats[f"{actor_str}_{conv_str}_played"]
        played_correct = stats[f"{actor_str}_{conv_str}_played_correct"]
        played_incorrect = stats[f"{actor_str}_{conv_str}_played_incorrect"]

        played_correct_available_percent = percent(played_correct, available)
        played_correct_played_percent = percent(played_correct, played)

        stats[f"{actor_str}_{conv_str}_played_correct_available%"] = \
                played_correct_available_percent
        stats[f"{actor_str}_{conv_str}_played_correct_played%"] = \
                played_correct_played_percent


def record_percent(stats, player, move_type, move_total):
    stat = stats[f"actor{player}_{move_type}"]
    stat_total = stats[f"actor{player}_{move_total}"]
    stats[f"actor{player}_{move_type}%"] = percent(stat, stat_total)

def percent(n, total):
    if total == 0:
        return 0
    return (n / total)
