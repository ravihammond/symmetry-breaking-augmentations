#!/usr/bin/env bash

docker run --rm -it \
    --gpus all \
    --ipc host \
    --env WANDB_TOKEN=$(cat wandb_api_key.txt) \
    ravihammond/hanabi-project:prod \
    scripts/iql_convention.sh 
