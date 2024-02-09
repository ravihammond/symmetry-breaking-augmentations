#!/bin/bash

# SAD ======================================

## BR_IQL_six vs IQL
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model iql_aht \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_IQL_six vs IQL
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model iql_aht \
#    --split_type six \
#    --split_indices 0-9 \

# BR_IQL_six vs SAD
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model sad \
#    --split_type six \
#    --split_indices 0 \

## SBA_IQL_six vs SAD
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model sad \
#    --split_type six \
#    --split_indices 0-9 \

## BR_IQL_six vs OP
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model op \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_IQL_six vs OP
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model op \
#    --split_type six \
#    --split_indices 0-9 \

## BR_IQL_six vs obl
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model obl \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_IQL_six vs OBL
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model obl \
#    --split_type six \
#    --split_indices 0-9 \

# IQL ======================================

# BR_IQL_one vs IQL
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model iql_aht \
    --split_type one \
    --split_indices 0-11 \

# SBA_IQL_one vs IQL
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model iql_aht \
    --split_type one \
    --split_indices 0-11 \

## BR_IQL_six vs IQL
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model iql_aht \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_IQL_six vs IQL
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model iql_aht \
#    --split_type six \
#    --split_indices 0-9 \

# BR_IQL_ten vs IQL
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model iql_aht \
    --split_type ten \
    --split_indices 0-9 \

# SBA_IQL_ten vs IQL
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model iql_aht \
    --split_type ten \
    --split_indices 0-9 \

# BR_IQL_one vs SAD
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model sad \
    --split_type one \
    --split_indices 0-11 \

# SBA_IQL_one vs SAD
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model sad \
    --split_type one \
    --split_indices 0-11 \

# BR_IQL_six vs SAD
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model sad \
#    --split_type six \
#    --split_indices 0 \

## SBA_IQL_six vs SAD
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model sad \
#    --split_type six \
#    --split_indices 0-9 \

# BR_IQL_ten vs SAD
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model sad \
    --split_type ten \
    --split_indices 0-9 \

# SBA_IQL_ten vs SAD
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model sad \
    --split_type ten \
    --split_indices 0-9 \

# BR_IQL_one vs OP
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model op \
    --split_type one \
    --split_indices 0-11 \

# SBA_IQL_one vs OP
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model op \
    --split_type one \
    --split_indices 0-11 \

## BR_IQL_six vs OP
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model op \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_IQL_six vs OP
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model op \
#    --split_type six \
#    --split_indices 0-9 \

# BR_IQL_ten vs OP
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model op \
    --split_type ten \
    --split_indices 0-9 \

# SBA_IQL_ten vs OP
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model op \
    --split_type ten \
    --split_indices 0-9 \

# BR_IQL_one vs obl
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model obl \
    --split_type one \
    --split_indices 0-11 \

# SBA_IQL_one vs OBL
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model obl \
    --split_type one \
    --split_indices 0-11 \

## BR_IQL_six vs obl
#python tools/aht_eval.py \
#    --aht_model br_iql \
#    --train_partner_model iql \
#    --partner_model obl \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_IQL_six vs OBL
#python tools/aht_eval.py \
#    --aht_model sba_iql \
#    --train_partner_model iql \
#    --partner_model obl \
#    --split_type six \
#    --split_indices 0-9 \

# BR_IQL_ten vs obl
python tools/aht_eval.py \
    --aht_model br_iql \
    --train_partner_model iql \
    --partner_model obl \
    --split_type ten \
    --split_indices 0-9 \

# SBA_IQL_ten vs OBL
python tools/aht_eval.py \
    --aht_model sba_iql \
    --train_partner_model iql \
    --partner_model obl \
    --split_type ten \
    --split_indices 0-9 \

# OP ======================================

## BR_OP_six vs OP
#python tools/aht_eval.py \
#    --aht_model br_op \
#    --train_partner_model op \
#    --partner_model op_aht \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_OP_six vs OP
#python tools/aht_eval.py \
#    --aht_model sba_op \
#    --train_partner_model op \
#    --partner_model op_aht \
#    --split_type six \
#    --split_indices 0-9 \

## BR_OP_six vs SAD
#python tools/aht_eval.py \
#    --aht_model br_op \
#    --train_partner_model op \
#    --partner_model sad \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_OP_six vs SAD
#python tools/aht_eval.py \
#    --aht_model sba_op \
#    --train_partner_model op \
#    --partner_model sad \
#    --split_type six \
#    --split_indices 0-9 \

## BR_OP_six vs IQL
#python tools/aht_eval.py \
#    --aht_model br_op \
#    --train_partner_model op \
#    --partner_model iql \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_OP_six vs IQL
#python tools/aht_eval.py \
#    --aht_model sba_op \
#    --train_partner_model op \
#    --partner_model iql \
#    --split_type six \
#    --split_indices 0-9 \

## BR_OP_six vs OBL
#python tools/aht_eval.py \
#    --aht_model br_op \
#    --train_partner_model op \
#    --partner_model obl \
#    --split_type six \
#    --split_indices 0-9 \

## SBA_OP_six vs OBL
#python tools/aht_eval.py \
#    --aht_model sba_op \
#    --train_partner_model op \
#    --partner_model obl \
#    --split_type six \
#    --split_indices 0-9 \
