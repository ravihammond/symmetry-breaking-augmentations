import numpy as np


def collect_stats(actors, scores, conventions):
    stats = {}

    get_scores(stats, scores)

    convention_strs = convention_strings(conventions)

    for i, actor in enumerate(actors):
        convention_index = actor.get_convention_index()
        convention_str = convention_strs[convention_index]

        actor_stats = actor.get_stats()

        record_actor_stats(stats, actor_stats, convention_str, i % 2)

    return stats


def get_scores(stats, scores):
    non_zero_scores = [s for s in scores if s > 0]
    non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
    bomb_out_rate = 100 * (1 - len(non_zero_scores) / len(scores))

    stats["non_zero_mean"] = non_zero_mean
    stats["bom_out_rate"] = bomb_out_rate


def convention_strings(conventions):
    convention_strs = []

    for convention in conventions:
        convention_str = ""
        for i, two_step in enumerate(convention):
            convention_str += two_step[0] + '-' + two_step[1]
            if i != len(convention) - 1:
                convention_str + '_'
        convention_strs.append(convention_str)

    return convention_strs


def record_actor_stats(stats, actor_stats, convention_str, player):
    move_stats(stats, actor_stats, convention_str, player)
    convention_stats(stats, actor_stats, convention_str, player)

 
def move_stats(stats, actor_stats, convention_str, player):
    pass


def convention_stats(stats, actor_stats, convention_str, player):
    pass




