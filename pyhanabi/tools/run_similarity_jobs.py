import os
import sys
import argparse
import pprint
pprint = pprint.pprint
import numpy as np
import json
from easydict import EasyDict as edict
import copy
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import subprocess
from functools import partial


def run_similarity_jobs(args):
    jobs = create_similarity_jobs(args)
    run_jobs(args, jobs)


def create_similarity_jobs(args):
    indexes = [[int(y) for y in x.split("-")] 
            for x in args.indexes.split(",")]

    jobs = []

    for i, (index_1, index_2) in enumerate(indexes):
        job = edict()

        job.device = f"cuda:{i}"
        (job.policy1, job.sad_legacy1, job.name1) = model_to_weight(
                args, args.model1, index_1)
        (job.policy2, job.sad_legacy2, job.name2) = model_to_weight(
                args, args.model2, index_2)

        jobs.append(job)

    return jobs


def parse_numbers(numbers):
    if '-' in numbers:
        range = [int(x) for x in numbers.split('-')]
        assert(len(range) == 2)
        numbers_list = list(np.arange(*range))
    else:
        numbers_list = [int(x) for x in numbers.split(',')]

    return numbers_list


def model_to_weight(args, model, model_index):
    player_name = model

    if model == "br":
        splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")
        indexes = splits[args.split_index]["train"]
        indexes = [x + 1 for x in indexes]
        indexes_str = '_'.join(str(x) for x in indexes)
        player_name = f"br_sad_{args.split_type}_{indexes_str}"
        path = f"../models/my_models/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0

    elif model == "sba":
        splits = load_json_list(f"train_test_splits/sad_splits_{args.split_type}.json")
        indexes = splits[args.split_index]["train"]
        indexes = [x + 1 for x in indexes]
        indexes_str = '_'.join(str(x) for x in indexes)
        player_name = f"sba_sad_{args.split_type}_{indexes_str}"
        path = f"../models/my_models/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0

    elif "obl" in model:
        policies = load_json_list("agent_groups/all_obl.json")
        path = policies[model_index]
        player_name = f"obl_{partner_index + 1}"
        sad_legacy = 0

    elif model == "op":
        policies = load_json_list("agent_groups/all_op.json")
        path = policies[model_index]
        player_name = f"op_{model_index + 1}"
        sad_legacy = 1

    elif model == "sad":
        policies = load_json_list("agent_groups/all_sad.json")
        path = policies[model_index]
        player_name = f"sad_{model_index + 1}"
        sad_legacy = 1

    return path, sad_legacy, player_name


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


def run_jobs(args, jobs):
    worker = partial(run_jobs_worker, args=args)
    num_workers = len(args.indexes.split(","))
    with ThreadPoolExecutor(num_workers) as pool:
        pool.map(worker, jobs)


def run_jobs_worker(job, args):
    try:
        command = ["python", "similarity.py",
            "--outdir", args.outdir,
            "--policy1", job.policy1,
            "--policy2", job.policy2,
            "--sad_legacy1", str(job.sad_legacy1),
            "--sad_legacy2", str(job.sad_legacy2),
            "--name1", job.name1,
            "--name2", job.name2,
            "--model1", args.model1,
            "--model2", args.model2,
            "--shuffle_index", args.shuffle_index,
            "--device", job.device,
            "--num_game", str(args.num_game),
            "--num_thread", str(args.num_thread),
            "--seed", str(args.seed),
            "--verbose", str(args.verbose),
            "--save", str(args.save),
            "--upload_gcloud", str(args.upload_gcloud),
            "--gcloud_dir", str(args.gcloud_dir),
        ]
        subprocess.run(command)
    except Exception as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="sad")
    parser.add_argument("--model2", type=str, default="sad")
    parser.add_argument("--indexes", type=str, default="0-1")
    parser.add_argument("--shuffle_index", type=str, default="0-120")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--upload_gcloud", type=int, default=1)
    parser.add_argument("--gcloud_dir", type=str, default="hanabi-similarity-new")
    parser.add_argument("--outdir", type=str, default="similarity")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_similarity_jobs(args)
