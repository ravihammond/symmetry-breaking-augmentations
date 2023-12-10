#!/bin/bash

# SAD ======================================

## BR_SAD_one vs SAD
#python tools/collect_action_stats.py \
#    --aht_model br_sad \
#    --train_partner_model sad \
#    --partner_model sad_aht \
#    --split_type one \
#    --split_indices 0-12 \
#    --save 1 \

## BR_SAD_six vs SAD
#python tools/collect_action_stats.py \
#    --aht_model br_sad \
#    --train_partner_model sad \
#    --partner_model sad_aht \
#    --split_type six \
#    --split_indices 0-9 \
#    --save 1 \

## BR_SAD_eleven vs SAD
#python tools/collect_action_stats.py \
#    --aht_model br_sad \
#    --train_partner_model sad \
#    --partner_model sad_aht \
#    --split_type eleven \
#    --split_indices 0-9 \
#    --save 1 \

## BR_IQL_six vs IQL
#python tools/collect_action_stats.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model iql_aht \
#    --split_type six \
#    --split_indices 0-9 \
#    --save 1 \

## BR_OP_six vs OP
#python tools/collect_action_stats.py \
#    --aht_model br_op \
#    --train_partner_model op \
#    --partner_model op_aht \
#    --split_type six \
#    --split_indices 0-9 \
#    --save 1 \

# SBA_SAD_one vs SAD
python tools/collect_action_stats.py \
    --aht_model sba_sad \
    --train_partner_model sad \
    --partner_model sad_aht \
    --split_type one \
    --split_indices 0-12 \
    --save 1 \

# SBA_SAD_six vs SAD
python tools/collect_action_stats.py \
    --aht_model sba_sad \
    --train_partner_model sad \
    --partner_model sad_aht \
    --split_type six \
    --split_indices 0-9 \
    --save 1 \

# SBA_SAD_eleven vs SAD
python tools/collect_action_stats.py \
    --aht_model sba_sad \
    --train_partner_model sad \
    --partner_model sad_aht \
    --split_type eleven \
    --split_indices 0-9 \
    --save 1 \

# SBA_IQL_six vs IQL
python tools/collect_action_stats.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model iql_aht \
    --split_type six \
    --split_indices 0-9 \
    --save 1 \

# SBA_OP_six vs OP
python tools/collect_action_stats.py \
    --aht_model sba_op \
    --train_partner_model op \
    --partner_model op_aht \
    --split_type six \
    --split_indices 0-9 \
    --save 1 \

