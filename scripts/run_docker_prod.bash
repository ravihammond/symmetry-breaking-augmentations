#!/usr/bin/env bash

docker run --rm -it \
    --gpus all \
    --env WANDB_TOKEN=$(cat wandb_api_key.txt) \
    ravihammond/hanabi-project:prod \
    exps/test1 \
    conventions/CR-P0.json
