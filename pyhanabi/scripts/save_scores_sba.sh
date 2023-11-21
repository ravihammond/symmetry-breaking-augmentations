#!/bin/bash

#python tools/save_scores_sba.py \
#    --outdir similarity \
#    --policy1 ../training_models/sad_2p_models/sad_1.pthw \
#    --policy2 ../training_models/sad_2p_models/sad_2.pthw \
#    --model1 sad \
#    --model2 sad \
#    --sad_legacy1 1 \
#    --sad_legacy2 1 \
#    --name1 sad_1 \
#    --name2 sad_2 \
#    --model1 sad \
#    --model1 sad \
#    --shuffle_index 100 \
#    --num_game 1 \

python tools/save_scores_sba.py \
    --outdir similarity \
    --policy1 ch_temp/sad_2p_models/sad_1.pthw \
    --policy2 ../training_models/sad_2p_models/sad_2.pthw \
    --model1 sad \
    --model2 sad \
    --sad_legacy1 1 \
    --sad_legacy2 1 \
    --name1 sad_1 \
    --name2 sad_2 \
    --model1 sad \
    --model1 sad \
    --shuffle_index 100 \
    --num_game 1 \
