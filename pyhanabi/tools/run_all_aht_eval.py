import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy
from datetime import datetime
import numpy as np
import random
import subprocess

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)


SPLIT_TYPES = {
    "sad": ["one", "six", "eleven"],
    # "iql": ["one", "six", "ten"],
    "iql": ["one", "ten"],
    "op": ["one", "six", "ten"],
}

SPLIT_INDICES = {
    "sad": {
        "one": "0-12", 
        "six": "0-9", 
        "eleven": "0-9"
    },
    "iql": {
        "one": "0-11", 
        "six": "0-9", 
        "ten": "0-9"
    },
    "op": {
        "one": "0-11", 
        "six": "0-9", 
        "ten": "0-9"
    }
}

def run_all_aht_eval(args):
    jobs = create_jobs(args)
    run_jobs(args, jobs)


def create_jobs(args):
    jobs = []

    for aht_model in args.aht_models.split(","):
        for split_type in SPLIT_TYPES[aht_model]:
            for partner_model in args.partner_models.split(","):
                if partner_model == aht_model:
                    partner_model += "_aht"
                for method in args.methods.split(","):
                    job = edict()
                    job.aht_model = f"{method}_{aht_model}"
                    job.train_partner_model = aht_model
                    job.partner_model = partner_model
                    job.split_type = split_type
                    job.split_indices = SPLIT_INDICES[aht_model][split_type]
                    jobs.append(job)

    return jobs


def run_jobs(args, jobs):
    for job in jobs:
        subprocess.run(["python", "tools/aht_eval.py",
            "--aht_model", job.aht_model,
            "--train_partner_model", job.train_partner_model,
            "--partner_model", job.partner_model,
            "--split_type", job.split_type,
            "--split_indices", job.split_indices,
            "--out", args.out,
            "--save", f"{args.save}",
        ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aht_models", type=str, default="sad,iql,op")
    parser.add_argument("--partner_models", type=str, default="sad,iql,op,obl")
    parser.add_argument("--methods", type=str, default="br,sba")
    parser.add_argument("--out", type=str, default="aht_scores")
    parser.add_argument("--save", type=int, default=1)
    return  parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_aht_eval(args)

