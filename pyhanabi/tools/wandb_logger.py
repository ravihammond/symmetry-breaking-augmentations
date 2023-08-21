import wandb
import numpy as np
import pprint
pprint = pprint.pprint

from tools.collect_actor_stats import collect_stats


def log_wandb_test(
        train_score, 
        train_perfect, 
        train_scores, 
        train_eval_actors, 
        loss_mean,
        aux_loss_mean,
        aux_accuracy_mean,
        aux_accuracy_per_step_mean,
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
        stat_type="train"
    )
    stats = collect_stats(
        test_score, 
        test_perfect, 
        test_scores, 
        test_eval_actors, 
        conventions,
        stat_type="test",
        stats=stats,
    )

    stats["loss"] = loss_mean
    stats["aux_loss"] = aux_loss_mean
    stats["aux_accuracy"] = aux_accuracy_mean
    for i, accuracy in enumerate(aux_accuracy_per_step_mean):
        stats[f"aux_accuracy_step_{i + 1}"] = accuracy

    wandb.log(dict(stats))

def log_wandb(
        train_score, 
        train_perfect, 
        train_scores, 
        train_eval_actors, 
        loss_mean,
        aux_loss_mean,
        aux_accuracy_mean,
        aux_accuracy_per_step_mean,
        conventions):

    stats = collect_stats(
        train_score, 
        train_perfect, 
        train_scores, 
        train_eval_actors, 
        conventions,
    )

    stats["loss"] = last_loss
    stats["aux_loss"] = last_aux_loss

    wandb.log(dict(stats))
