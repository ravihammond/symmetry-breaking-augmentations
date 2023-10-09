import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy
from datetime import datetime

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from tools.eval_model_verbose import evaluate_model


def save_scores_sba(args):
    jobs = create_jobs(args)
    print(jobs)

    run_jobs(jobs)


def create_jobs(args):
    jobs = []
    shuffle_indexes = parse_numbers(args.shuffle_index)

    job = edict()

    for shuffle_index in shuffle_indexes:
        job2 = copy.copy(job)
        job2.permute_index = shuffle_index
        jobs.append(job2)

    return jobs


def parse_numbers(numbers):
    if '-' in numbers:
        range = [int(x) for x in numbers.split('-')]
        assert(len(range) == 2)
        numbers_list = list(np.arange(*range))
    else:
        numbers_list = [int(x) for x in numbers.split(',')]

    return numbers_list


def run_jobs(jobs):
    for job in jobs:
        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y_%H:%M:%S")
        job.csv_name = f"{job.csv_name}_{date_time}.csv"
        input_args = edict({
            "weight1": job.model1,
            "weight2": job.model2,
            "csv_name": job.csv_name,
            "sad_legacy": job.sad_legacy,
            "num_game": args.num_game,
            # other needed
            "weight3": None,
            "output": None,
            "num_player": 2,
            "seed": 0,
            "bomb": 0,
            "num_run": 1,
            "device": "cuda:0",
            "convention": "None",
            "convention_index": None,
            "override0": 0,
            "override1": 0,
            "belief_stats": 0,
            "belief_model": "None",
            "partner_models": "None",
            "train_test_splits": None,
            "split_index": 0,
        })
        evaluate_model(input_args)
        print()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--sad_legacy1", type=int, default=0)
    parser.add_argument("--sad_legacy2", type=int, default=0)
    parser.add_argument("--shuffle_index", type=str, default="0")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    save_scores_sba(args)

