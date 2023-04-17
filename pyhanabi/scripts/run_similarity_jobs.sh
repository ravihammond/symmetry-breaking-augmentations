#!/bin/bash

python tools/run_similarity_jobs.py \
    --model sad \
    --model_index 0 \
    --shuffle_index 0-120 \
    --workers 10 \
    --num_game 1000 \
    --upload_gcloud 0 \
    --seed 0 \
    --verbose 1 \
    --save 0 \
    --upload_gcloud 0 \
