import wandb
import pprint
import numpy as np
pprint = pprint.pprint

def log_wandb(score, perfect, scores, actors, loss):
    # wandb.log({
        # "score": score,
        # "perfect": perfect,
        # "loss": loss,
    # })

    # wandb.log(scores(scores))
    # wandb.log(actor_stats(actors, 0))
    # wandb.log(actor_stats(actors, 1))

    pprint({
        "score": score,
        "perfect": perfect,
        "loss": loss,
    })
    pprint(get_scores(scores))
    pprint(actor_stats(actors, 0))
    pprint(actor_stats(actors, 1))


def get_scores(scores):
    non_zero_scores = [s for s in scores if s > 0]
    non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
    bomb_out_rate = 100 * (1 - len(non_zero_scores) / len(scores))

    return ({
        "non_zero_mean": non_zero_mean, 
        "bomb_out_rate": bomb_out_rate
    })


def actor_stats(actors, player):
    return merge_dictionaries([
        played_card_knowledge(actors, player),
        move_stats(actors, player),
        convention_stats(actors, player),
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


def convention_stats(actors, player):
    available = sum_stats("convention_available", actors, player)
    played = sum_stats("convention_played", actors, player)
    played_correct = sum_stats("convention_played_correct", actors, player)
    played_incorrect = sum_stats("convention_played_incorrect", actors, player)
    played_correct_available_percent = percent(played_correct, available)
    played_correct_played_percent = percent(played_correct, played)

    stats = {
        f"actor{player}_convention_available": int(available),
        f"actor{player}_convention_played": played,
        f"actor{player}_convention_played_correct": played_correct,
        f"actor{player}_convention_played_correct_available%": \
            played_correct_available_percent,
        f"actor{player}_convention_played_correct_played%": \
            played_correct_played_percent,
        f"actor{player}_convention_played_incorrect": played_incorrect
    }

    for i in range(5):
        playable = sum_stats(f"convention_played_{i}_playable", actors, player)
        stats[f"actor{player}_convention_played_{i}_playable"] = playable

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


