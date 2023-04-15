#!/bin/bash

python similarity.py \
    --outdir test \
    --policy1 ../training_models/sad_2p_models/sad_1.pthw \
    --sad_legacy1 1 \
    --shuffle_colour1 1 \
    --permute_index1 100 \
    --name1 sad_1 \
    --policy2 ../training_models/sad_2p_models/sad_2.pthw \
    --sad_legacy2 1 \
    --shuffle_colour2 0 \
    --permute_index2 0 \
    --name2 sad_2 \
    --num_game 1000 \
    --num_thread 10 \
    --seed 0 \
    --verbose 1 \
    --save 1 \
    --upload_gcloud 1 \
