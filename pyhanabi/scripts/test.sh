#!/bin/bash

python selfplay.py \
       --save_dir exps/test \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 2254257 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 1000 \
       --num_epoch 2000 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --convention_act_override 1 \
       --convention conventions/hint_red_play_0.json 
       #--partner_model exps/obl1/model_epoch1400.pthw \
       #--static_partner 0 \
