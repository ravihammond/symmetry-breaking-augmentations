#!/bin/bash
python selfplay.py \
       --save_dir exps/pobl2_CR-P1_CB-P1 \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 2254257 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 3001 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 1 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2 \
       --num_lstm_layer 2 \
       --boltzmann_act 0 \
       --min_t 0.01 \
       --max_t 0.1 \
       --off_belief 1 \
       --num_fict_sample 10 \
       --belief_device cuda:3 \
       --belief_model exps/pbelief_obl1f_CR-P0/model0.pthw \
       --load_model None \
       --net publ-lstm \
       --convention conventions/all_colours.json \
       --num_conventions 25 \
       --parameterized 1 \
       --wandb 1 \
