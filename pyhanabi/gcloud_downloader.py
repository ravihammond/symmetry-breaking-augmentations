import os
import sys
import time
import argparse
from collections import defaultdict
import pprint
pprint = pprint.pprint

from google.cloud import storage

PROJECT = "aiml-reid-research"
GCLOUD_PATH = "Ravi/hanabi-conventions/"

def download_from_gcloud(args):
    client = storage.Client(project=PROJECT)
    bucket = client.get_bucket(PROJECT + "-data")

    prefix = os.path.join(GCLOUD_PATH, args.folder)
    all_blobs = list(client.list_blobs(bucket, prefix=prefix))
    assert len(all_blobs) != 0, f"Google Cloud Error: Prefix {prefix} doesn't exist."

    blobs = defaultdict(list)

    for blob in all_blobs:
        start_index = blob.name[:len(prefix)].rfind(args.folder)
        filepath = blob.name[start_index:]
        path = os.path.dirname(filepath)
        out_path = os.path.join(args.out, path)
        blobs[out_path].append(blob)

    blobs = dict(blobs)

    for outdir, blob_list in blobs.items():
        gcloud_dir = GCLOUD_PATH + os.path.basename(outdir)
        print(f"downloading: {gcloud_dir} -> {outdir}")
        if os.path.exists(outdir):
            sys.exit(f"Error: local directory already exists: {outdir}")
        else:
            os.makedirs(outdir)

        for blob in blob_list:
            filename = os.path.basename(blob.name)
            filepath = os.path.join(outdir, filename)

            accepted = ["train.log", "model_epoch1000.pthw"]
            accepted_file = any([True for s in accepted if s in filepath])
            if not args.last_only or accepted_file:
                blob.download_to_filename(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--rename", type=str, default=None)
    parser.add_argument("--out", type=str, default="temp")
    parser.add_argument('--last_only', type=int, default=1)
    args = parser.parse_args()
    download_from_gcloud(args)
