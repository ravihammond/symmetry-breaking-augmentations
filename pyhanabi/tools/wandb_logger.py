import wandb
import pprint
import numpy as np
pprint = pprint.pprint

# from collect_actor_stats import collect_stats

def log_wandb(score, perfect, scores, actors, loss, convention):
    # wandb.log({
        # "score": score,
        # "perfect": perfect,
        # "loss": loss,
    # })
    # wandb.log(get_scores(scores))
    # wandb.log(actor_stats(actors, 0))
    # wandb.log(actor_stats(actors, 1))
    pprint(get_scores(scores))
    pprint(actor_stats(actors, 0, convention))
    pprint(actor_stats(actors, 1, convention))


def get_scores(scores):
    non_zero_scores = [s for s in scores if s > 0]
    non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
    bomb_out_rate = 100 * (1 - len(non_zero_scores) / len(scores))

    return ({
        "non_zero_mean": non_zero_mean, 
        "bomb_out_rate": bomb_out_rate
    })


def actor_stats(actors, player, convention):
    return merge_dictionaries([
        played_card_knowledge(actors, player),
        move_stats(actors, player),
        convention_stats(actors, player, convention),
    ])


def played_card_knowledge(actors, player):
    card_stats = []
    for i, g in enumerate(actors):
        if i % 2 == player:
            card_stats.append(g.get_played_card_info())
    card_stats = np.array(card_stats).sum(0)
    total_played = sum(card_stats)

    stats = {}
    stats[f"actor{player}_total_cards_played"] = total_played

    for i, ck in enumerate(["none", "color", "rank", "both"]):
        percentage = (card_stats[i] / total_played) * 100
        percentage = percent(card_stats[i], total_played)
        stats[f"actor{player}_card_played_knowledge_{ck}%"] = percentage

    return stats


def move_stats(actors, player):
    colour_move_map = ["red", "yellow", "green", "white", "blue"]
    rank_move_map = ["1", "2", "3", "4", "5"]

    return merge_dictionaries([
        move_type_stat(actors, player, "play"),
        move_type_stat(actors, player, "discard"),
        move_type_stat(actors, player, "hint_colour"),
        move_type_stats(actors, player, "hint", colour_move_map, "hint_colour"),
        move_type_stat(actors, player, "hint_rank"),
        move_type_stats(actors, player, "hint", rank_move_map, "hint_rank"),
    ])


def move_type_stats(actors, player, move_type, move_map, move_total):
    total = sum_stats(move_total, actors, player)
    stats = {}

    for move in move_map:
        move_total = sum_stats(move_type + "_" + move, actors, player)
        stats[f"actor{player}_{move_type}_{move}%"] = \
                percent(move_total, total)

    return stats


def convention_stats(actors, player, conventions):
    convention_strings = []
    for convention_set in conventions:
        convention_string = ""
        for i, convention in enumerate(convention_set):
            if i > 0:
                convention_string += "-"
            convention_string += convention[0] + convention[1]
        convention_strings.append(convention_string)

    stats = {}

    for convention_string in convention_strings:
        conv_str = "convention_" + convention_string
        available = sum_stats(f"{conv_str}_available", actors, player)
        played = sum_stats(f"{conv_str}_played", actors, player)
        played_correct = sum_stats(f"{conv_str}_played_correct", actors, player)
        played_incorrect = sum_stats(f"{conv_str}_played_incorrect", actors, player)
        played_correct_available_percent = percent(played_correct, available)
        played_correct_played_percent = percent(played_correct, played)

        stats[f"actor{player}_{conv_str}_available"] = int(available)
        stats[f"actor{player}_{conv_str}_played"] = played
        stats[f"actor{player}_{conv_str}_played_correct"] = played_correct
        stats[f"actor{player}_{conv_str}_played_correct_available%"] = \
                played_correct_available_percent
        stats[f"actor{player}_{conv_str}_played_correct_played%"] = \
                played_correct_played_percent
        stats[f"actor{player}_{conv_str}_played_incorrect"] = played_incorrect

    return stats


def move_type_stat(actors, player, move_type):
    count = sum_stats(move_type, actors, player)
    return {f"actor{player}_{move_type}": int(count)}


def sum_stats(key, actors, player):
    stats = []
    for i, g in enumerate(actors): 
        if i % 2 == player:
            if key in g.get_stats():
                stats.append(g.get_stats()[key])
    return int(sum(stats))


def percent(n, total):
    if total == 0:
        return 0
    return (n / total)


def merge_dictionaries(dicts):
    stats = {}
    for dic in dicts:
        for key, value in dic.items():
            stats[key] = value
    return stats

