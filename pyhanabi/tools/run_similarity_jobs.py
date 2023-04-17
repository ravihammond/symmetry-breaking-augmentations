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


def run_similarity_jobs(args):
    jobs = create_similarity_jobs(args)
    run_jobs(args, jobs)


def create_similarity_jobs(args):
    jobs = []
    shuffle_indexes = parse_numbers(args.shuffle_index)

    models = load_json_list(f"agent_groups/all_{args.model}.json")
    job = edict()
    job.outdir = os.path.join(args.outdir, args.model)

    job.policy1, job.sad_legacy1, job.name1 = model_to_weight(
            args, args.model, args.model_index)
    job.shuffle_colour1 = 1

    for j in range(args.model_index, len(models)):
        if args.model_index == j:
            continue
        job2 = copy.copy(job)

        job2.policy2, job2.sad_legacy2, job2.name2 = model_to_weight(
                args, args.model, j)
        job2.shuffle_colour2 = 0
        job2.permute_index2 = 0
        for shuffle_index in shuffle_indexes:
            job3 = copy.copy(job2)
            job3.permute_index1 = shuffle_index

            jobs.append(job3)

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


class DeviceID:
    def __init__(self, number, max):
        self._number = number
        self._max = max

    def get(self):
        return self._number

    def set(self, number):
        self._number = number

    def increment(self):
        self._number = (self._number + 1) % self._max


def run_jobs(args, jobs):
    device_id_wrapper = DeviceID(0, 3)
    device_mutex = Lock()
    executor = ThreadPoolExecutor(args.workers)
    executor.map(run_jobs_worker, 
            jobs, [device_id_wrapper] * len(jobs), 
            [device_mutex] * len(jobs))


def run_jobs_worker(job, device_id_wrapper, device_mutex):
    try:
        with device_mutex:
            device_id = device_id_wrapper.get()
            device_id_wrapper.increment()
            device = f"cuda:{device_id + 1}"
        command = ["python", "similarity.py",
            "--outdir", job.outdir,
            "--policy1", job.policy1,
            "--sad_legacy1", str(job.sad_legacy1),
            "--shuffle_colour1", str(job.shuffle_colour1),
            "--permute_index1", str(job.permute_index1),
            "--name1", job.name1,
            "--policy2", job.policy2,
            "--sad_legacy2", str(job.sad_legacy2),
            "--shuffle_colour2", str(job.shuffle_colour2),
            "--permute_index2", str(job.permute_index2),
            "--name2", job.name2,
            "--device", device,
            "--num_game", str(args.num_game),
            "--num_thread", "10",
            "--seed", "0",
            "--verbose", str(args.verbose),
            "--save", str(args.save),
            "--upload_gcloud", str(args.upload_gcloud),
        ]
        subprocess.run(command)
    except Exception as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="br")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--shuffle_index", type=str, default="0")
    parser.add_argument("--seed", type=str, default="0")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--upload_gcloud", type=int, default=1)
    parser.add_argument("--gcloud_dir", type=str, default="hanabi-search-games-sba")
    parser.add_argument("--outdir", type=str, default="similarity")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_similarity_jobs(args)
