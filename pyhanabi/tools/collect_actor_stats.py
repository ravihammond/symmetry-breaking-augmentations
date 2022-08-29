import numpy as np
from collections import defaultdict
import pprint
pprint = pprint.pprint

CARD_INDEX_MAP = ["0", "1", "2", "3", "4"]
COLOUR_MOVE_MAP = ["red", "yellow", "green", "white", "blue"]
RANK_MOVE_MAP = ["1", "2", "3", "4", "5"]

def collect_stats(score, perfect, scores, actors, conventions):
    stats = defaultdict(int)
    record_total_scores(stats, score, perfect, scores)
    convention_strings = extract_convention_strings(conventions)

    convention_scores = defaultdict(list)

    for i, actor in enumerate(actors):
        actor_stats = defaultdict(int, actor.get_stats())
        move_stats(stats, actor_stats, i % 2)

        if len(convention_strings) == 0:
            continue

        convention_index = actor.get_convention_index()
        convention_str = convention_strings[convention_index]
        record_actor_stats(stats, actor_stats, convention_str, i % 2)

        if i % 2 == 0:
            convention_scores[convention_str].append(scores[i // 2])

    calculate_scores(stats, convention_strings, convention_scores)
    evaluate_percentages(stats, convention_strings)
    return dict(stats)


def record_total_scores(stats, score, perfect, scores):
    non_zero_scores = [s for s in scores if s > 0]
    non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
    bomb_out_rate = (1 - len(non_zero_scores) / len(scores))

    stats["score"] = score
    stats["perfect"] = perfect
    stats["non_zero_mean"] = non_zero_mean
    stats["bomb_out_rate"] = bomb_out_rate

def calculate_scores(stats, conventions, convention_scores):
    for convention in conventions:
        scores = convention_scores[convention]
        score = np.mean(scores)
        num_perfect = sum([1 for s in scores if s == 25])
        perfect = num_perfect / len(scores)
        non_zero_scores = [s for s in scores if s > 0]
        non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
        bomb_out_rate = (1 - len(non_zero_scores) / len(scores))

        stats[f"{convention}_score"] = score
        stats[f"{convention}_perfect"] = perfect
        stats[f"{convention}_non_zero_mean"] = non_zero_mean
        stats[f"{convention}_bomb_out_rate"] = bomb_out_rate


def extract_convention_strings(conventions):
    convention_strings = []

    for convention in conventions:
        convention_str = ""
        for i, two_step in enumerate(convention):
            if i > 0:
                convention_str + '-'
            convention_str += two_step[0] + two_step[1]
        convention_strings.append(convention_str)

    return convention_strings


def record_actor_stats(stats, actor_stats, convention_str, player):
    move_stats(stats, actor_stats, player, convention_str)
    convention_stats(stats, actor_stats, player, convention_str, "signal")
    convention_stats(stats, actor_stats, player, convention_str, "response")

 
def move_stats(stats, actor_stats, player, convention=None):
    move_type_stats(stats, actor_stats, player, 
            CARD_INDEX_MAP, "play", convention)
    move_type_stats(stats, actor_stats, player, 
            CARD_INDEX_MAP, "discard", convention)
    move_type_stats(stats, actor_stats, player, 
            COLOUR_MOVE_MAP, "hint", convention, "_colour")
    move_type_stats(stats, actor_stats, player, 
            RANK_MOVE_MAP, "hint", convention, "_rank")


def move_type_stats(stats, actor_stats, player, move_map, move_type, 
        convention, move_type_suffix=""):
    prefix = f"actor{player}"
    if convention != None:
        prefix = convention + "_" + prefix
    move_with_suffix = f"{move_type}{move_type_suffix}"

    stats[f"{prefix}_{move_with_suffix}"] += int(actor_stats[move_with_suffix])

    for move in move_map:
        stats[f"{prefix}_{move_type}_{move}"] += \
                int(actor_stats[f"{move_type}_{move}"])


def convention_stats(stats, actor_stats, player, convention, role):
    prefix = f"{convention}_actor{player}_{role}"
    convention_str = f"{role}_{convention}"
    available = f"{convention_str}_available"
    played = f"{convention_str}_played"
    played_correct = f"{convention_str}_played_correct"
    played_incorrect = f"{convention_str}_played_incorrect"

    stats[f"{prefix}_available"] += int(actor_stats[available])
    stats[f"{prefix}_played"] += int(actor_stats[played])
    stats[f"{prefix}_played_correct"] += int(actor_stats[played_correct])
    stats[f"{prefix}_played_incorrect"] += int(actor_stats[played_incorrect])


def evaluate_percentages(stats, conventions):
    for player in range(2):
        move_percentages(stats, player)
    for convention in conventions:
        for player in range(2):
            move_percentages(stats, player, convention)
            convention_percentages(stats, player, convention, "signal")
            convention_percentages(stats, player, convention, "response")

def move_percentages(stats, player, convention=None):
    percent_move_type(stats, player, CARD_INDEX_MAP, "play", convention)
    percent_move_type(stats, player, CARD_INDEX_MAP, "discard", convention)
    percent_move_type(stats, player, COLOUR_MOVE_MAP, "hint", convention, "_colour")
    percent_move_type(stats, player, RANK_MOVE_MAP, "hint", convention, "_rank")

def percent_move_type(stats, player, move_map, move_type, 
        convention, move_type_suffix=""):
    actor_str = f"actor{player}"
    if convention != None:
        actor_str = convention + "_" + actor_str
    move_with_suffix = f"{move_type}{move_type_suffix}"

    total = stats[f"{actor_str}_{move_with_suffix}"]

    for move in move_map:
        move_type_with_move = f"{actor_str}_{move_type}_{move}"
        move_count = stats[move_type_with_move]
        percentage = percent(move_count, total)
        stats[f"{move_type_with_move}%"] = percentage

def convention_percentages(stats, player, convention, role):
    prefix = f"{convention}_actor{player}_{role}"
    available = stats[f"{prefix}_available"]
    played = stats[f"{prefix}_played"]
    played_correct = stats[f"{prefix}_played_correct"]
    played_incorrect = stats[f"{prefix}_played_incorrect"]

    played_correct_available_percent = percent(played_correct, available)
    played_correct_played_percent = percent(played_correct, played)

    stats[f"{prefix}_played_correct_available%"] = \
            played_correct_available_percent
    stats[f"{prefix}_played_correct_played%"] = \
            played_correct_played_percent


def record_percent(stats, player, move_type, move_total):
    stat = stats[f"actor{player}_{move_type}"]
    stat_total = stats[f"actor{player}_{move_total}"]
    stats[f"actor{player}_{move_type}%"] = percent(stat, stat_total)

def percent(n, total):
    if total == 0:
        return 0
    return (n / total)
