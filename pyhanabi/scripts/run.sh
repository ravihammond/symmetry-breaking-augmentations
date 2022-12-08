#!/bin/bash

if [ ! -z "$WANDB_TOKEN" ]
then
    wandb login $WANDB_TOKEN
else 
    echo "Exiting, WANDB_TOKEN env var not set."
    exit 128
fi

exec $@
