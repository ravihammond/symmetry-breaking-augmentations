#!/bin/bash

python selfplay.py \
       --save_dir exps/br_aux_sad_four \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed $1 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 1000 \
       --num_epoch 10001 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --train_partner_models agent_groups/all_sad.json \
       --train_partner_sad_legacy 1 \
       --train_test_splits train_test_splits/sad_splits_four.json \
       --split_index $1 \
       --static_partner 1 \
       --save_checkpoints 10 \
       --class_aux_weight $2 \
       --wandb 1 \
       --gcloud_upload 1

