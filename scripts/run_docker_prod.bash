#!/usr/bin/env bash

GCLOUD_PATH="/home/ravi/.config/gcloud/application_default_credentials.json"

docker run --rm -it\
    --gpus all \
    --ipc host \
    --volume=${GCLOUD_PATH}:/gcloud_creds.json:rw \
    --env GOOGLE_APPLICATION_CREDENTIALS=/gcloud_creds.json \
    --env WANDB_TOKEN=$(cat wandb_api_key.txt) \
    ravihammond/hanabi-project:prod \
    ${@:-bash}
    #scripts/$(basename -- $1) 
