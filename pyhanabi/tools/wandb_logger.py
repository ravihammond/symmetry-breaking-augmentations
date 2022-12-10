import wandb
import pprint
import numpy as np
pprint = pprint.pprint

from tools.collect_actor_stats import collect_stats


def log_wandb(
        train_score, 
        train_perfect, 
        train_scores, 
        train_eval_actors, 
        last_loss, 
        test_score, 
        test_perfect, 
        test_scores, 
        test_eval_actors, 
        conventions):

    stats = collect_stats(
        train_score, 
        train_perfect, 
        train_scores, 
        train_eval_actors, 
        conventions,
        "train"
    )
    stats = collect_stats(
        test_score, 
        test_perfect, 
        test_scores, 
        test_eval_actors, 
        conventions,
        "test",
        stats=stats,
    )

    stats["loss"] = last_loss

    wandb.log(dict(stats))
