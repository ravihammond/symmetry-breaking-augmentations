#!/usr/bin/env bash

docker run --rm -it \
    --gpus all \
    --ipc host \
    ravihammond/hanabi-project:prod \
    ${@:-bash}

