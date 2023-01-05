#!/bin/bash

#python selfplay.py \
       #--save_dir exps/br_sad \
       #--num_thread 24 \
       #--num_game_per_thread 80 \
       #--method iql \
       #--sad 0 \
       #--lr 6.25e-05 \
       #--eps 1.5e-05 \
       #--gamma 0.999 \
       #--seed $1 \
       #--burn_in_frames 10000 \
       #--replay_buffer_size 100000 \
       #--batchsize 128 \
       #--epoch_len 1000 \
       #--num_epoch 1001 \
       #--num_player 2 \
       #--net lstm \
       #--num_lstm_layer 2 \
       #--multi_step 3 \
       #--train_device cuda:0 \
       #--act_device cuda:1,cuda:2,cuda:3 \
       #--partner_models agent_groups/all_sad.json \
       #--partner_sad_legacy 1 \
       #--train_test_splits sad_train_test_splits.json \
       #--split_index $1 \
       #--static_partner 1 \
       #--wandb 1 \
       #--gcloud_upload 1

if [ ! -z "$WANDB_TOKEN" ]
then
    wandb login $WANDB_TOKEN
else 
    echo "Exiting, WANDB_TOKEN env var not set."
    exit 128
fi

python selfplay.py \
       --save_dir exps/pp \
       --load_model exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 0 \
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
       --test_partner_models agent_groups/all_sad.json \
       --test_partner_sad_legacy 1 \
       --train_test_splits sad_train_test_splits.json \
       --split_index 0 \
       --wandb 1 \
       --gcloud_upload 0 \
       #--train_partner_model exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw \
       #--static_partner 1 \
