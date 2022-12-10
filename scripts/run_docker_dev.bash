#!/usr/bin/env bash

GCLOUD_PATH="/home/ravi/.config/gcloud/application_default_credentials.json"

docker run --rm -it \
    --volume=$(pwd):/app/:rw \
    --gpus all \
    --ipc host \
    --volume=${GCLOUD_PATH}:/gcloud_creds.json:rw \
    --env GOOGLE_APPLICATION_CREDENTIALS=/gcloud_creds.json \
    --env WANDB_TOKEN=$(cat keys/wandb_api_key.txt) \
    ravihammond/hanabi-project:dev \
    ${@:-bash}
