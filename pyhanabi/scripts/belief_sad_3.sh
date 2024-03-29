#!/bin/bash
python train_belief.py \
       --save_dir exps/belief0_sad_3.sh \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --seed 987389232 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --hid_dim 512 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 5001 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --explore 1 \
       --num_player 2 \
       --policy ../models/sad_2p_models/sad_3.pthw \
       --sad_legacy 1 \

#python train_belief.py \
       #--save_dir exps/test \
       #--num_thread 24 \
       #--num_game_per_thread 80 \
       #--batchsize 50 \
       #--lr 6.25e-05 \
       #--eps 1.5e-05 \
       #--grad_clip 5 \
       #--hid_dim 512 \
       #--burn_in_frames 100 \
       #--replay_buffer_size 1000 \
       #--epoch_len 2 \
       #--num_epoch 2 \
       #--train_device cuda:0 \
       #--act_device cuda:1 \
       #--policy ../models/sad_models/sad_2p_3.pthw \
       #--sad_legacy 1 \
       #--seed 1298329 \
       #--num_player 2 \
