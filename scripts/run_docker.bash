#!/usr/bin/env bash

docker run --rm -it \
    --volume=$(pwd):/app/:rw \
    --gpus all \
    --ipc host \
    --env WANDB_TOKEN=$(cat wandb_api_key.txt) \
    ravihammond/obl-project \
    ${@:-bash}

