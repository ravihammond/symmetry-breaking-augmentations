import os
import sys
import time
import argparse
from collections import defaultdict
import pprint
pprint = pprint.pprint
import numpy as np
from google.cloud import storage
import json
import pathlib

PROJECT = "aiml-reid-research"
GCLOUD_PATH = "Ravi"
SPLIT_NAME = { "six": "6-7-splits", "one": "1-12-splits" }
NUM_PERMUTES = 120

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def check_similarity(args):
    model_weights1 = load_json_list(f"agent_groups/all_{args.model1}.json")
    model_weights2 = load_json_list(f"agent_groups/all_{args.model2}.json")

    xp_indexes = get_crossplay_indexes(args, model_weights1, model_weights2)

    client = storage.Client(project=PROJECT)
    bucket = client.get_bucket(PROJECT + "-data")

    for idx1, idx2 in xp_indexes:
        check_files(args, idx1, idx2, client, bucket)


def get_crossplay_indexes(args, model_weights1, model_weights2):
    xp_indexes = []
    for i in range(len(model_weights1)):
        start_j = 0
        if args.model1 == args.model2:
            start_j = i
        for j in range(start_j, len(model_weights2)):
            if args.model1 == args.model2 and i == j: 
                continue
            xp_indexes.append((i + 1, j + 1))
    return xp_indexes


def check_files(args, idx1, idx2, client, bucket):
    name1 = f"{args.model1}_{idx1}"
    name2 = f"{args.model2}_{idx2}"
    print(f"{name1} vs {name2}")

    prefix = os.path.join(
        GCLOUD_PATH, 
        args.dir,
        f"{args.model1}_vs_{args.model2}",
        f"{name1}_vs_{name2}",
    )

    check_all_files(args, prefix, "similarity", client, bucket)
    check_all_files(args, prefix, "scores", client, bucket)


def check_all_files(args, prefix, data_type, client, bucket):
    full_prefix = os.path.join(prefix, data_type)
    all_blobs = list(client.list_blobs(bucket, prefix=full_prefix))

    if (len(all_blobs) == 0):
        print(bcolors.FAIL + f"No {data_type} found." + bcolors.ENDC)
        return

    blobs = []
    filepaths = []

    for blob in all_blobs:
        filepaths.append(blob.name)
        blobs.append(blob)

    filepaths_str = "\t".join(filepaths)
    missing = []
    found = []
    missing_permutes = []

    for permute_idx in range(NUM_PERMUTES):
        permute_str = f"permute_{permute_idx}."
        if permute_str not in filepaths_str:
            missing.append(permute_str)
            missing_permutes.append(permute_idx)
        else:
            found.append(permute_str)

    if len(missing) == 0:
        print(bcolors.OKGREEN + f"All {data_type} found." + bcolors.ENDC)
        download_bloblist(args, client, bucket, blobs)
    elif len(missing) == NUM_PERMUTES:
        print(bcolors.FAIL + f"No {data_type} found." + bcolors.ENDC)
    else:
        print(bcolors.WARNING + f"Partial {data_type} found, {len(missing)} left.")
        if args.verbose:
            for i, missing_permute in enumerate(missing_permutes):
                print(f"{missing_permute}", end="")
                if i < len(missing_permutes) - 1:
                    print(",", end="")
            print()
        print(bcolors.ENDC, end="")


def download_bloblist(args, client, bucket, blobs):
    for blob in blobs:
        blob_path_obj = pathlib.Path(blob.name)
        truncated_blob_path = os.path.join(*blob_path_obj.parts[2:])
        output_path = os.path.join(args.out, truncated_blob_path)

        output_dir_path = os.path.dirname(output_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        blob.download_to_filename(output_path)


def load_json_list(convention_path):
    if convention_path == "None":
        return []
    convention_file = open(convention_path)
    return json.load(convention_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="sad")
    parser.add_argument("--model2", type=str, default="sad")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--dir", type=str, default="hanabi-similarity")
    parser.add_argument("--out", type=str, default="similarity")
    args = parser.parse_args()

    check_similarity(args)


