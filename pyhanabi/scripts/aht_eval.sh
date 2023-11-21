#!/bin/bash

## SBA
#python tools/aht_eval.py \
#    --aht_model sba_sad \
#    --train_partner_model sad \
#    --partner_model sad_aht \
#    --split_type six \
#    --split_indices 0-9 \
#    --num_game 1000

## CH SBA
#python tools/aht_eval.py \
#    --aht_model ch_e50_sba_sad \
#    --train_partner_model sad \
#    --partner_model sad_aht \
#    --split_type six \
#    --split_indices 0-9 \
#    --num_game 1000

## SBA
#python tools/aht_eval.py \
#    --aht_model br_sad \
#    --train_partner_model sad \
#    --partner_model sad_aht \
#    --split_type six \
#    --split_indices 0-9 \
#    --num_game 1000

# CH SBA
python tools/aht_eval.py \
    --aht_model ch_e50_sad \
    --train_partner_model sad \
    --partner_model sad_aht \
    --split_type six \
    --split_indices 0-9 \
    --num_game 1000
