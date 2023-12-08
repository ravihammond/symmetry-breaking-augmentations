#!/bin/bash

#python selfplay.py \
#       --save_dir exps/hsba_sad \
#       --num_thread 24 \
#       --num_game_per_thread 80 \
#       --method iql \
#       --sad 0 \
#       --lr 6.25e-05 \
#       --eps 1.5e-05 \
#       --gamma 0.999 \
#       --seed $1 \
#       --burn_in_frames 10000 \
#       --replay_buffer_size 100000 \
#       --batchsize 128 \
#       --epoch_len 1000 \
#       --num_epoch 10001 \
#       --num_player 2 \
#       --net lstm \
#       --num_lstm_layer 2 \
#       --multi_step 3 \
#       --train_device cuda:0 \
#       --act_device cuda:1,cuda:2,cuda:3 \
#       --train_partner_models agent_groups/all_sad.json \
#       --train_partner_sad_legacy 1 \
#       --train_test_splits train_test_splits/sad_splits_six.json \
#       --split_index $1 \
#       --static_partner 1 \
#       --shuffle_color 1 \
#       --save_checkpoints 10 \
#       --wandb 1 \
#       --gcloud_upload 1
       
python selfplay.py \
       --save_dir exps/test \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 0 \
       --burn_in_frames 100 \
       --replay_buffer_size 10000 \
       --batchsize 10 \
       --epoch_len 10 \
       --num_epoch 100 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --train_partner_models agent_groups/all_sad.json \
       --train_partner_sad_legacy 1 \
       --train_test_splits train_test_splits/sad_splits_six.json \
       --split_index 0 \
       --static_partner 1 \
       --shuffle_color 1 \
       --dist_shuffle_colour 1 \
       --wandb 0 \
       --gcloud_upload 0
