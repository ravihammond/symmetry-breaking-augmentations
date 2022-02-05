#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/app/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    -p 8888:8888 \
    --gpus all \
    --ipc host \
    ravihammond/obl-project \
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
