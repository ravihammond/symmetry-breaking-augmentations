#!/bin/bash

python train_belief.py \
       --save_dir exps/belief_sba_sad \
       --num_thread 26 \
       --num_game_per_thread 80 \
       --seed $1 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --hid_dim 512 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 1001 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --explore 1 \
       --num_player 2 \
       --sad_legacy 1 \
       --runner_div round_robin \
       --policy_list agent_groups/all_sad.json \
       --train_test_splits sad_train_test_splits.json \
       --split_index $1 \
       --shuffle_color 1 \
       --wandb 1 \
       --gcloud_upload 1

#python train_belief.py \
       #--save_dir exps/belief_sba_sad \
       #--num_thread 1 \
       #--num_game_per_thread 1 \
       #--seed 0 \
       #--lr 6.25e-05 \
       #--eps 1.5e-05 \
       #--grad_clip 5 \
       #--hid_dim 512 \
       #--batchsize 1 \
       #--burn_in_frames 10 \
       #--replay_buffer_size 10 \
       #--epoch_len 10 \
       #--num_epoch 100 \
       #--train_device cuda:0 \
       #--act_device cuda:1,cuda:2,cuda:3 \
       #--explore 1 \
       #--num_player 2 \
       #--sad_legacy 1 \
       #--runner_div round_robin \
       #--policy_list agent_groups/all_sad.json \
       #--train_test_splits sad_train_test_splits.json \
       #--split_index 0 \
       #--shuffle_color 1 \
       #--wandb 0 \
       #--gcloud_upload 0
