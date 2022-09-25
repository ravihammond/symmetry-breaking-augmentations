#!/bin/bash
if [ ! -z "$WANDB_TOKEN" ]
then
    wandb login $WANDB_TOKEN
else 
    echo "Exiting, WANDB_TOKEN env var not set."
    exit 128
fi

python selfplay.py \
       --save_dir $2 \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 2254257 \
       --burn_in_frames 1 \
       --replay_buffer_size 10 \
       --batchsize 1 \
       --epoch_len 1000 \
       --num_epoch 1 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:0 \
       --convention $3 \
       --convention_act_override 3 \
       --partner_model ../training_models/obl1/model0.pthw \
       --static_partner 1 \
       --wandb 0 \
