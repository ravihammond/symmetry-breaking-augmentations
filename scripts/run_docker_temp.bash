#!/usr/bin/env bash

docker run --rm -it \
    --volume=$(pwd):/app/:rw \
    --gpus all \
    --ipc host \
    ravihammond/obl-project-temp \
    ${@:-bash}

