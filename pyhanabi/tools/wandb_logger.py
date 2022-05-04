import wandb
import pprint
import numpy as np
pprint = pprint.pprint

from tools.collect_actor_stats import collect_stats

def log_wandb(score, perfect, scores, actors, loss, conventions):
    stats = collect_stats(score, perfect, scores, actors, conventions)
    wandb.log({
        "score": score,
        "perfect": perfect,
        "loss": loss,
    })
    wandb.log(stats)
