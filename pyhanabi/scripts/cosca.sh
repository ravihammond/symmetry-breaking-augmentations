#!/bin/bash

#python selfplay.py \
       #--save_dir exps/iql_obl1_CR-P1_CB-P1_r \
       #--num_thread 24 \
       #--num_game_per_thread 80 \
       #--method iql \
       #--sad 0 \
       #--lr 6.25e-05 \
       #--eps 1.5e-05 \
       #--gamma 0.999 \
       #--seed 2254257 \
       #--burn_in_frames 10000 \
       #--replay_buffer_size 100000 \
       #--batchsize 128 \
       #--epoch_len 1000 \
       #--num_epoch 3001 \
       #--num_player 2 \
       #--net lstm \
       #--num_lstm_layer 2 \
       #--multi_step 3 \
       #--train_device cuda:0 \
       #--act_device cuda:1,cuda:2,cuda:3 \
       #--convention conventions/CR-P1_CB-P1.json \
       #--convention_act_override 2 \
       #--partner_models agent_groups/all_sad.json \
       #--static_partner 1 \
       #--wandb 1 \

       #--partner_model ../training_models/obl1/model0.pthw \

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
       --batchsize 1 \
       --epoch_len 1 \
       --num_epoch 2 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --convention conventions/CR-P0_CR-P1_CR-P2_CR-P3_CR-P4.json \
       --convention_act_override 2 \
       --partner_models agent_groups/all_sad.json \
       --static_partner 1 \
       --wandb 0 \

       #--partner_model ../training_models/obl1/model0.pthw \

