import pprint
pprint = pprint.pprint
from collections import defaultdict

COLOURS = "RYGWB"
RANKS = "12345"
CARDS = "01234"

ACTION_MAP = {
    "C": COLOURS,
    "R": RANKS,
    "P": CARDS,
    "D": CARDS
}

def extract_convention_stats(actors, args, convention_strings=[]):
    action_counts = defaultdict(int)

    for i, actor in enumerate(actors):
        convention_index = actor.get_convention_index()
        convention_str = None
        if len(convention_strings) > 0:
            convention_str = convention_strings[convention_index]
        actor_stats = defaultdict(int, actor.get_stats())
        record_action_counts(action_counts, actor_stats, convention_str, i % 2)

    stats = calculate_plot_stats(action_counts, convention_strings, args.split)

    return stats


def record_action_counts(action_counts, actor_stats, convention, player_idx):
    for signal_type in "DPCR":
        signal_response_counts(action_counts, actor_stats, 
                convention, signal_type, player_idx)


def signal_response_counts(action_counts, actor_stats, 
        convention, signal_type, player_idx):
    type_map = ACTION_MAP[signal_type]

    for s_idx in range(5):
        signal = f"{signal_type}{type_map[s_idx]}"
        for r_idx in range(5):
            for response_type in "DPCR":
                response_counts(action_counts, actor_stats, 
                        convention, signal, response_type, r_idx, player_idx)


def response_counts(action_counts, actor_stats, 
        convention, signal, response_type, index, player_idx):
    type_map = ACTION_MAP[response_type]
    two_step = f"{signal}_{response_type}{type_map[index]}"
    if convention is not None:
        action_counts[f"{convention}:{player_idx}:{two_step}"] += \
                actor_stats[two_step]
    action_counts[f"{player_idx}:{two_step}"] += actor_stats[two_step]


def calculate_plot_stats(action_counts, conventions, split):
    stats = []

    stats.append(action_matrix_stats(action_counts, None, split))

    for convention in conventions:
        stats.append(action_matrix_stats(action_counts, convention, split))
    
    return stats


def action_matrix_stats(action_counts, convention, split):
    action_matrices = {}

    title = "All"
    if convention is not None:
        title = convention
    action_matrices["title"] = title

    plots = []

    for player_idx in range(2 if split else 1):
        stats = {}
        for signal_type in "DPCR":
            signal_response_stats(stats, action_counts, convention, 
                    signal_type, player_idx, split)
        plots.append(stats)

    action_matrices["plots"] = plots

    return action_matrices


def signal_response_stats(stats, action_counts, convention, 
        signal_type, player_idx, split):
    for s_action in ACTION_MAP[signal_type]:
        signal = f"{signal_type}{s_action}"

        total = get_signal_total(action_counts, convention, signal, 
                                 player_idx, split)

        response_stats(stats, action_counts, convention, signal, total,
                       player_idx, split)

def get_signal_total(action_counts, convention, signal, player_idx, split):
    total = 0

    prefix = "" if convention is None else convention + ":"

    for r_type in "DPCR":
        for r_action in ACTION_MAP[r_type]:
            two_step = f"{signal}_{r_type}{r_action}"

            if split:
                key = f"{prefix}{player_idx}:{two_step}"
                total += action_counts[key]
                continue

            for p in range(2):
                key = f"{prefix}{p}:{two_step}"
                total += action_counts[key]

    return total

def response_stats(stats, action_counts, convention, signal, 
        total, player_idx, split):
    prefix = "" if convention is None else convention + ":"

    for r_type in "CDPR":
        for r_action in ACTION_MAP[r_type]:
            two_step = f"{signal}_{r_type}{r_action}"
            key = f"{prefix}{player_idx}:{two_step}"

            if split: 
                count = action_counts[key]
                stats[key] = divide(count, total)
                continue

            count = 0
            for p in range(2):
                k = f"{prefix}{p}:{two_step}"
                count += action_counts[k]

            stats[key] = divide(count, total)


def divide(n, total):
    if total == 0:
        return 0
    return (n / total)


