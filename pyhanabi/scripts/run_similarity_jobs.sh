#!/bin/bash

#python tools/run_similarity_jobs.py \
    #--model1 sad \
    #--model2 sad \
    #--indexes 0-1,0-2,0-3,0-4 \
    #--shuffle_index 0-3 \
    #--upload_gcloud 0 \
    #--verbose 0 \
    #--save 0 \
    #--upload_gcloud 0 \
    #--num_thread 10 \

python tools/run_similarity_jobs.py \
    --model1 op \
    --model2 op \
    --indexes 0-1 \
    --shuffle_index 0 \
    --upload_gcloud 0 \
    --verbose 1 \
    --save 1 \
    --upload_gcloud 0 \
    --num_thread 10 \
