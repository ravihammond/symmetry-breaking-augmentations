#!/bin/bash

if [ ! -z "$WANDB_TOKEN" ]
then
    wandb login $WANDB_TOKEN
else 
    exit 128
fi

python selfplay.py \
       --save_dir $1 \
       --num_thread 10 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 2254257 \
       --burn_in_frames 1 \
       --replay_buffer_size 100000 \
       --batchsize 1 \
       --epoch_len 1000 \
       --num_epoch 1 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --convention_act_override 0 \
       --convention $2 \
       --partner_model ../training_models/obl1/model0.pthw \
       --static_partner 1 \
       --wandb 1

#python selfplay.py \
       #--save_dir exps/test \
       #--num_thread 24 \
       #--num_game_per_thread 80 \
       #--method iql \
       #--sad 0 \
       #--lr 6.25e-05 \
       #--eps 1.5e-05 \
       #--gamma 0.999 \
       #--seed 2254257 \
       #--burn_in_frames 1 \
       #--replay_buffer_size 100000 \
       #--batchsize 1 \
       #--epoch_len 1000 \
       #--num_epoch 1 \
       #--num_player 2 \
       #--net lstm \
       #--num_lstm_layer 2 \
       #--multi_step 3 \
       #--train_device cuda:0 \
       #--act_device cuda:1 \
       #--convention_act_override 0 \
       #--convention conventions/CR-P0.json \
       #--partner_model ../training_models/obl1/model0.pthw \
       #--static_partner 1 \
       #--wandb 1
