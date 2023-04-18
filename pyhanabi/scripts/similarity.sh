#!/bin/bash

python similarity.py \
    --outdir similarity/sad \
    --policy1 ../training_models/sad_2p_models/sad_1.pthw \
    --policy2 ../training_models/sad_2p_models/sad_2.pthw \
    --sad_legacy1 1 \
    --sad_legacy2 1 \
    --name1 sad_1 \
    --name2 sad_2 \
    --shuffle_index 0-100 \
    --num_game 10000 \
    --num_thread 10 \
    --seed 0 \
    --verbose 1 \
    --save 0 \
    --upload_gcloud 0 \
    --workers 1 \
    --device cuda:2,cuda:3 \
