#!/bin/bash
python selfplay.py \
       --save_dir exps/test \
       --num_thread 1 \
       --num_game_per_thread 10 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 2254257 \
       --burn_in_frames 10 \
       --replay_buffer_size 100 \
       --batchsize 10 \
       --epoch_len 1000 \
       --num_epoch 101 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --wandb 0 \
       --google_cloud_upload 1 \
       --save_checkpoints 5 \

